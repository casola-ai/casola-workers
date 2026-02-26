"""
Queue transport abstraction for Casola GPU workers.

Provides a unified interface for workers to receive jobs and send results
via WebSocket push (queue).

Usage:
    from casola_worker.transport import create_transport

    transport = create_transport(config)
    transport.on_job = my_process_function
    transport.start()  # blocking
"""

import json
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base
# =============================================================================


class QueueTransport(ABC):
    """Abstract base for job transport implementations."""

    on_job: Optional[Callable[[Dict[str, Any]], None]] = None
    on_connected: Optional[Callable[[], None]] = None

    @abstractmethod
    def start(self) -> None:
        """Blocking job loop. Returns when stop() is called."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Signal the transport to shut down gracefully."""
        ...

    def drain(self) -> None:
        """Stop accepting new jobs but keep connection alive for in-flight results."""
        ...

    def pause(self) -> None:
        """Signal the server to stop dispatching new jobs (reversible via resume)."""
        ...

    def resume(self) -> None:
        """Signal the server to resume dispatching new jobs after a pause."""
        ...

    @abstractmethod
    def complete_job(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Report successful job completion. Returns True on success."""
        ...

    @abstractmethod
    def fail_job(self, job_id: str, error: str) -> bool:
        """Report job failure. Returns True on success."""
        ...

    @abstractmethod
    def send_progress(
        self,
        job_id: str,
        fence_token: int = 0,
        progress: Optional[float] = None,
        status_message: Optional[str] = None,
    ) -> bool:
        """Send progress / lease renewal. Returns True on success."""
        ...


# =============================================================================
# WebSocket Push Transport (queue)
# =============================================================================


class WebSocketTransport(QueueTransport):
    """Connects to a queue Durable Object via WebSocket for push-based dispatch."""

    # Reconnect backoff parameters
    INITIAL_BACKOFF_S = 1.0
    MAX_BACKOFF_S = 60.0
    BACKOFF_FACTOR = 2.0
    # Auto lease renewal for persisted jobs
    LEASE_RENEWAL_INTERVAL_S = 30

    def __init__(self, config: Dict[str, Any]):
        self.ws_url: str = config["ws_url"]
        self.worker_id: str = config["worker_id"]
        self.config_id: str = config["config_id"]
        self.capacity: int = config.get("capacity", 1)
        self.api_token: Optional[str] = config.get("api_token")

        self.running = False
        self.draining = False
        self.paused = False
        self._ws: Any = None  # websockets connection object
        self._ws_lock = threading.Lock()
        # Active persisted jobs: job_id -> fence_token
        # Active persisted jobs: job_id -> { fence_token, result_upload_url }
        self._active_persisted: Dict[str, Dict] = {}
        self._active_lock = threading.Lock()
        self._lease_thread: Optional[threading.Thread] = None
        self._lease_stop = threading.Event()

        self.OFFLOAD_THRESHOLD_BYTES = 256 * 1024

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        try:
            from websockets.sync.client import connect as ws_connect
        except ImportError:
            raise ImportError(
                "websockets package is required for WebSocket transport. "
                "Install it with: pip install websockets>=12.0"
            )

        self.running = True
        self._lease_stop.clear()
        self._start_lease_renewal_thread()

        backoff = self.INITIAL_BACKOFF_S
        while self.running:
            try:
                ws_endpoint = self.ws_url.rstrip("/") + "/ws"
                logger.info("WebSocketTransport: connecting to %s", ws_endpoint)
                extra_headers = {}
                if self.api_token:
                    extra_headers["Authorization"] = f"Bearer {self.api_token}"
                with ws_connect(
                    ws_endpoint, additional_headers=extra_headers, ping_interval=30, ping_timeout=10
                ) as ws:
                    with self._ws_lock:
                        self._ws = ws

                    # Send register message
                    register_msg = {
                        "type": "register",
                        "worker_id": self.worker_id,
                        "config_id": self.config_id,
                        "capacity": self.capacity,
                    }
                    ws.send(json.dumps(register_msg))
                    logger.info(
                        "WebSocketTransport: registered worker_id=%s, config_id=%s, capacity=%d",
                        self.worker_id,
                        self.config_id,
                        self.capacity,
                    )

                    # Reset backoff on successful connect
                    backoff = self.INITIAL_BACKOFF_S

                    # Notify caller that connection is ready
                    if self.on_connected:
                        self.on_connected()

                    # Receive loop
                    for raw in ws:
                        if not self.running:
                            break
                        self._handle_message(raw)

            except Exception:
                if not self.running:
                    break
                logger.exception("WebSocketTransport: connection error")

            with self._ws_lock:
                self._ws = None

            if not self.running:
                break

            # Reconnect with exponential backoff + jitter
            jitter = random.uniform(0, backoff * 0.3)
            sleep_time = backoff + jitter
            logger.info("WebSocketTransport: reconnecting in %.1fs", sleep_time)
            time.sleep(sleep_time)
            backoff = min(backoff * self.BACKOFF_FACTOR, self.MAX_BACKOFF_S)

        self._stop_lease_renewal_thread()
        logger.info("WebSocketTransport: stopped")

    def stop(self) -> None:
        self.running = False
        self._stop_lease_renewal_thread()
        with self._ws_lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass

    def drain(self) -> None:
        self.draining = True
        # Tell the DO to stop pushing new jobs (sets capacity to 0 server-side)
        self._ws_send({"type": "drain"})
        logger.info("WebSocketTransport: draining (notified server, new jobs will be rejected)")

    def pause(self) -> None:
        if self.paused:
            return
        self.paused = True
        self._ws_send({"type": "pause"})
        logger.info("WebSocketTransport: paused (notified server, new dispatches halted)")

    def resume(self) -> None:
        if not self.paused:
            return
        self.paused = False
        self._ws_send({"type": "resume"})
        logger.info("WebSocketTransport: resumed (notified server, dispatches restarted)")

    # -- job operations -------------------------------------------------------

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> bool:
        active = self._pop_active(job_id)
        fence_token = active.get("fence_token", 0)
        result_upload_url = active.get("result_upload_url")

        result_bytes = json.dumps(result).encode()
        if result_upload_url and len(result_bytes) > self.OFFLOAD_THRESHOLD_BYTES:
            # Upload result directly to R2 via presigned PUT URL
            try:
                import urllib.request as _urllib_request

                req = _urllib_request.Request(result_upload_url, data=result_bytes, method="PUT")
                req.add_header("Content-Type", "application/json")
                _urllib_request.urlopen(req, timeout=60)
                logger.info(
                    "WebSocketTransport: uploaded result for job %s (%d bytes) to R2",
                    job_id,
                    len(result_bytes),
                )
                # Notify DO with a ref signal only — no result bytes in WS message
                return self._ws_send(
                    {
                        "type": "job_result_ref",
                        "job_id": job_id,
                        "fence_token": fence_token,
                    }
                )
            except Exception:
                logger.exception(
                    "WebSocketTransport: R2 upload failed for job %s, falling back to inline",
                    job_id,
                )

        # Inline path: small result or upload failed
        msg = {
            "type": "job_result",
            "job_id": job_id,
            "result": result,
            "fence_token": fence_token,
        }
        return self._ws_send(msg)

    def fail_job(self, job_id: str, error: str) -> bool:
        active = self._pop_active(job_id)
        fence_token = active.get("fence_token", 0)
        msg = {
            "type": "job_error",
            "job_id": job_id,
            "error": error,
            "fence_token": fence_token,
        }
        return self._ws_send(msg)

    def send_progress(
        self,
        job_id: str,
        fence_token: int = 0,
        progress: Optional[float] = None,
        status_message: Optional[str] = None,
    ) -> bool:
        msg: Dict[str, Any] = {
            "type": "job_progress",
            "job_id": job_id,
            "fence_token": fence_token,
        }
        if progress is not None:
            msg["progress"] = progress
        if status_message is not None:
            msg["status_message"] = status_message
        return self._ws_send(msg)

    # -- message handling -----------------------------------------------------

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("WebSocketTransport: invalid JSON: %s", raw[:200])
            return

        msg_type = msg.get("type")

        if msg_type == "job":
            job_id = msg.get("job_id", "")
            mode = msg.get("mode", "transient")
            fence_token = msg.get("fence_token", 0)
            config_id = msg.get("config_id", "")
            payload_url = msg.get("payload_url")
            result_upload_url = msg.get("result_upload_url")

            # Reject new jobs while draining (shutting down)
            if self.draining:
                logger.info("WebSocketTransport: draining, rejecting job %s", job_id)
                return

            # Fetch payload from R2 if a presigned URL was provided
            if payload_url:
                try:
                    import urllib.request as _urllib_request

                    with _urllib_request.urlopen(payload_url, timeout=30) as resp:
                        payload = json.loads(resp.read())
                    logger.debug("WebSocketTransport: fetched payload for job %s from R2", job_id)
                except Exception:
                    logger.exception(
                        "WebSocketTransport: failed to fetch payload for job %s from R2", job_id
                    )
                    payload = msg.get("payload", {})
            else:
                payload = msg.get("payload", {})

            # Track persisted jobs for auto lease renewal and result upload
            if mode == "persisted":
                with self._active_lock:
                    self._active_persisted[job_id] = {
                        "fence_token": fence_token,
                        "result_upload_url": result_upload_url,
                    }

            # Build a job dict compatible with the worker's process_job()
            job = {
                "id": job_id,
                "config_id": config_id,
                "payload": payload,
                "mode": mode,
                "fence_token": fence_token,
            }

            if self.on_job:
                # Run in a thread so we can continue receiving messages
                thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
                thread.start()
            else:
                logger.warning("WebSocketTransport: on_job not set, dropping job %s", job_id)

        elif msg_type == "job_result_ref":
            # Server acknowledgement of a job_result_ref — nothing to do client-side
            pass

        elif msg_type == "job_revoked":
            job_id = msg.get("job_id", "")
            reason = msg.get("reason", "")
            logger.warning("WebSocketTransport: job %s revoked: %s", job_id, reason)
            with self._active_lock:
                self._active_persisted.pop(job_id, None)

        elif msg_type == "error":
            message = msg.get("message", "")
            logger.error("WebSocketTransport: server error: %s", message)

        else:
            logger.warning("WebSocketTransport: unknown message type: %s", msg_type)

    def _run_job(self, job: Dict[str, Any]) -> None:
        try:
            if self.on_job:
                self.on_job(job)
        except Exception:
            logger.exception("WebSocketTransport: on_job raised for job %s", job.get("id"))

    # -- auto lease renewal ---------------------------------------------------

    def _start_lease_renewal_thread(self) -> None:
        self._lease_stop.clear()
        self._lease_thread = threading.Thread(target=self._lease_renewal_loop, daemon=True)
        self._lease_thread.start()

    def _stop_lease_renewal_thread(self) -> None:
        self._lease_stop.set()
        if self._lease_thread:
            self._lease_thread.join(timeout=5)
            self._lease_thread = None

    def _lease_renewal_loop(self) -> None:
        while not self._lease_stop.is_set():
            if self._lease_stop.wait(timeout=self.LEASE_RENEWAL_INTERVAL_S):
                break
            # Send progress for all active persisted jobs
            with self._active_lock:
                active = dict(self._active_persisted)
            for job_id, info in active.items():
                fence_token = info.get("fence_token", 0) if isinstance(info, dict) else info
                self.send_progress(job_id, fence_token=fence_token, status_message="processing")

    # -- internal helpers -----------------------------------------------------

    def _pop_active(self, job_id: str) -> Dict:
        """Remove a job from the active set, returning its tracking dict."""
        with self._active_lock:
            return self._active_persisted.pop(job_id, {})

    def _ws_send(self, msg: Dict[str, Any]) -> bool:
        with self._ws_lock:
            if self._ws is None:
                logger.error("WebSocketTransport: not connected, cannot send %s", msg.get("type"))
                return False
            try:
                self._ws.send(json.dumps(msg))
                return True
            except Exception:
                logger.exception("WebSocketTransport: send failed")
                return False


# =============================================================================
# Factory
# =============================================================================


def create_transport(config: Dict[str, Any]) -> QueueTransport:
    """Create a WebSocket transport for connecting to queue."""
    return WebSocketTransport(config)
