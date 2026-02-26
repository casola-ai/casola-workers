"""
System metrics collection and shipping for GPU workers.

Collects GPU metrics (via nvidia-smi), engine-specific metrics (via callback),
and worker metadata. Ships batches to the Queue's POST /metrics endpoint.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

log = logging.getLogger(__name__)


def collect_gpu_metrics() -> Dict[str, float]:
    """Collect GPU metrics via nvidia-smi. Returns dict of available metrics."""
    if not shutil.which("nvidia-smi"):
        return {}

    fields = ["memory.used", "memory.total", "utilization.gpu", "temperature.gpu"]

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, Exception):
        return {}

    if output.returncode != 0:
        return {}

    metrics: Dict[str, float] = {}
    # Parse first GPU line only
    for line in output.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(fields):
            break

        def _parse(val: str) -> Optional[float]:
            try:
                v = val.strip()
                if v in ("[N/A]", "N/A", ""):
                    return None
                return float(v)
            except (ValueError, TypeError):
                return None

        mem_used = _parse(parts[0])
        mem_total = _parse(parts[1])
        util = _parse(parts[2])
        temp = _parse(parts[3])

        if mem_used is not None:
            metrics["gpu_memory_used_mb"] = mem_used
        if mem_total is not None:
            metrics["gpu_memory_total_mb"] = mem_total
        if util is not None:
            metrics["gpu_utilization_pct"] = util
        if temp is not None:
            metrics["gpu_temperature_c"] = temp
        break  # Only first GPU

    return metrics


class SystemMetricsReporter:
    """Collects and ships system metrics to the Queue's POST /metrics endpoint."""

    def __init__(
        self,
        instance_id: str,
        queue_id: str,
        config_id: str,
        api_token: str,
        interval: int = 30,
        engine_metrics_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        log_fn: Optional[Callable[..., None]] = None,
    ):
        self.instance_id = instance_id
        self.queue_id = queue_id
        self.config_id = config_id
        self.api_token = api_token
        self.interval = interval
        self.engine_metrics_fn = engine_metrics_fn
        self.log_fn = log_fn or (lambda msg, **kw: log.info(msg))
        self.metrics_url = os.environ.get("CASOLA_METRICS_URL", "")
        self.running = False
        self._start_time = time.time()
        self._current_job_id: Optional[str] = None
        self._job_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def set_current_job(self, job_id: Optional[str]) -> None:
        """Track the currently running job (thread-safe)."""
        with self._job_lock:
            self._current_job_id = job_id

    def collect_snapshot(self) -> Dict[str, Any]:
        """Collect a single metrics snapshot."""
        snapshot: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "worker_uptime_s": round(time.time() - self._start_time, 1),
        }

        # GPU metrics
        gpu = collect_gpu_metrics()
        snapshot.update(gpu)

        # Engine-specific metrics
        if self.engine_metrics_fn:
            try:
                engine = self.engine_metrics_fn()
                snapshot.update(engine)
            except Exception:
                pass  # Engine metrics are best-effort

        # Current job
        with self._job_lock:
            if self._current_job_id:
                snapshot["current_job_id"] = self._current_job_id

        return snapshot

    def ship_metrics(self, snapshots: List[Dict[str, Any]]) -> None:
        """POST metrics batch to the Queue. No-op if URL not configured."""
        if not self.metrics_url or not self.api_token:
            return

        payload = {
            "instance_id": self.instance_id,
            "queue_id": self.queue_id,
            "config_id": self.config_id,
            "snapshots": snapshots,
        }

        try:
            resp = requests.post(
                self.metrics_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_token}",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                self.log_fn(f"Metrics ship failed: HTTP {resp.status_code}")
        except Exception as e:
            self.log_fn(f"Metrics ship error: {e}", level="warning")

    def _metrics_loop(self) -> None:
        """Background loop: collect + ship metrics at interval."""
        while self.running:
            time.sleep(self.interval)
            if not self.running:
                break
            try:
                snapshot = self.collect_snapshot()
                self.ship_metrics([snapshot])
            except Exception as e:
                self.log_fn(f"Metrics collection error: {e}", level="warning")

    def start(self) -> None:
        """Start the metrics collection thread."""
        self.running = True
        self._thread = threading.Thread(target=self._metrics_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the metrics collection thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def flush(self) -> None:
        """Collect and ship a final snapshot (for shutdown)."""
        try:
            snapshot = self.collect_snapshot()
            self.ship_metrics([snapshot])
        except Exception:
            pass
