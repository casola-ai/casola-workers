#!/usr/bin/env python3

import base64
import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any

import requests

# Allow importing casola_worker from parent directory (local dev)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from casola_worker.config import build_transport_config
from casola_worker.error_patterns import check_log_for_errors, compile_error_patterns
from casola_worker.system_metrics import SystemMetricsReporter
from casola_worker.transport import QueueTransport, create_transport

# LightX2V-specific error patterns (appended to the shared base lists)
_LIGHTX2V_FATAL_EXTRA = [
    r"forward compatibility was attempted on non supported HW",
    r"unsupported display driver / cuda driver combination",
]

_LIGHTX2V_RETRYABLE_EXTRA = [
    r"RuntimeError.*initialization failed",
    r"Failed to load model",
    r"Model .* does not exist",
    r"Failed to initialize",
    r"Error loading model weights",
]


class LightX2VWorker:
    def __init__(self):
        self.api_url = os.environ.get("CASOLA_API_URL")
        self.api_token = os.environ.get("CASOLA_API_TOKEN")

        container_id = os.environ.get("CONTAINER_ID", str(os.getpid()))
        self.instance_id = os.environ.get("CASOLA_INSTANCE_ID", f"vast-{container_id}")
        self.queue_id = os.environ.get("CASOLA_QUEUE_ID", "lightx2v-queue")
        self.worker_id = self.instance_id

        self.heartbeat_interval = int(os.environ.get("CASOLA_HEARTBEAT_INTERVAL", "60"))
        self.max_jobs = int(os.environ.get("CASOLA_MAX_JOBS", "1"))
        self.lease_seconds = int(os.environ.get("CASOLA_LEASE_SECONDS", "60"))
        self.shutdown_grace_period = float(os.environ.get("CASOLA_SHUTDOWN_GRACE_PERIOD", "5.0"))

        # Read config_id from environment (set by scheduler)
        self.config_id = os.environ.get("CASOLA_CONFIG_ID", "")
        if not self.config_id:
            raise ValueError("CASOLA_CONFIG_ID environment variable is required")

        # LightX2V-specific config
        self.model_path = os.environ.get("LIGHTX2V_MODEL_PATH", "")
        if not self.model_path:
            raise ValueError("LIGHTX2V_MODEL_PATH environment variable is required")
        self.model_cls = os.environ.get("LIGHTX2V_MODEL_CLS", "")
        if not self.model_cls:
            raise ValueError("LIGHTX2V_MODEL_CLS environment variable is required")
        self.config_json = os.environ.get("LIGHTX2V_CONFIG_JSON", "")
        if not self.config_json:
            raise ValueError("LIGHTX2V_CONFIG_JSON environment variable is required")

        self.lightx2v_task = os.environ.get("LIGHTX2V_TASK", "t2v")
        self.lightx2v_host = os.environ.get("LIGHTX2V_HOST", "127.0.0.1")
        self.lightx2v_port = os.environ.get("LIGHTX2V_PORT", "8000")
        self.lightx2v_url = f"http://{self.lightx2v_host}:{self.lightx2v_port}"
        self.lightx2v_startup_timeout = int(os.environ.get("LIGHTX2V_STARTUP_TIMEOUT", "600"))

        self.running = True
        self.last_heartbeat = 0
        self.current_job_id = None
        self.job_lock = threading.Lock()

        self.transport: QueueTransport | None = None

        self.engine_process = None
        self.engine_log_monitor_thread = None
        self.error_detected = False
        self.error_message = None
        self.error_is_fatal = False

        # Timing: record container startup
        self.container_start_time = time.time()
        self.engine_start_time: float | None = None
        created_at = os.environ.get("CASOLA_CREATED_AT")
        self.scheduler_created_at: float | None = float(created_at) if created_at else None

        # Compile regex patterns for efficiency
        self.compiled_fatal_patterns, self.compiled_retryable_patterns = compile_error_patterns(
            fatal_extra=_LIGHTX2V_FATAL_EXTRA,
            retryable_extra=_LIGHTX2V_RETRYABLE_EXTRA,
        )

        # Log shipping
        self.log_buffer: list[dict[str, Any]] = []
        self.log_lock = threading.Lock()
        self.last_log_flush = time.time()
        self.log_flush_interval = int(os.environ.get("CASOLA_LOG_FLUSH_INTERVAL", "30"))
        self.ws_url = os.environ.get("CASOLA_WS_URL", "")

        # System metrics reporter (initialized in run() after engine is ready)
        self.metrics_reporter: SystemMetricsReporter | None = None

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        self.log("Received shutdown signal, cleaning up...")
        self.running = False

        # Stop accepting new jobs but keep transport alive for in-flight results
        if self.transport:
            self.transport.drain()

        with self.job_lock:
            current_job = self.current_job_id

        if current_job:
            self.log(
                f"Waiting up to {self.shutdown_grace_period}s for job {current_job} to complete..."
            )
            start_time = time.time()

            while time.time() - start_time < self.shutdown_grace_period:
                with self.job_lock:
                    if self.current_job_id is None:
                        self.log("Job completed during grace period")
                        break
                time.sleep(0.1)
            else:
                with self.job_lock:
                    if self.current_job_id:
                        self.log(f"Grace period expired for job {self.current_job_id}")

        # Flush remaining logs before shutdown
        self.flush_logs()

        # Stop the transport after grace period so in-flight results can be delivered
        if self.transport:
            self.transport.stop()

        self.send_heartbeat("stopping")
        sys.exit(0)

    def log(
        self,
        message: str,
        level: str = "info",
        source: str = "worker",
        job_id: str | None = None,
        config_id: str | None = None,
    ):
        """Log a message from the worker or subprocess (LightX2V)."""
        timestamp_s = int(time.time())
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Always print to stdout for container logs
        print(f"{timestamp_str} - [{source.upper()}] [{level.upper()}] {message}", flush=True)

        # Add to buffer for shipping to platform
        entry = {
            "timestamp": timestamp_s,
            "level": level,
            "source": source,
            "message": message,
        }

        if job_id:
            entry["job_id"] = job_id
        if config_id:
            entry["config_id"] = config_id

        with self.log_lock:
            self.log_buffer.append(entry)

            # Prevent unbounded memory growth
            if len(self.log_buffer) > 10000:
                self.log_buffer = self.log_buffer[-5000:]

    def flush_logs(self):
        """Send buffered logs to Queue DO via HTTP."""
        with self.log_lock:
            if not self.log_buffer:
                return

            # Take up to 1000 logs
            batch = self.log_buffer[:1000]
            self.log_buffer = self.log_buffer[1000:]

        # Get logs URL from env var (injected by scheduler)
        logs_url = os.environ.get("CASOLA_LOGS_URL")
        if not logs_url or not self.api_token:
            return

        payload = {
            "instance_id": self.instance_id,
            "queue_id": self.queue_id,
            "entries": batch,
        }

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }

            resp = requests.post(logs_url, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()

        except Exception:
            # Don't crash worker if log shipping fails
            with self.log_lock:
                self.log_buffer = batch + self.log_buffer

                # Prevent unbounded growth
                if len(self.log_buffer) > 10000:
                    self.log_buffer = self.log_buffer[-10000:]

    def _log_flush_loop(self):
        """Background thread that flushes logs periodically."""
        while self.running:
            time.sleep(self.log_flush_interval)
            self.flush_logs()

    def print_environment_variables(self):
        """Print environment variables on startup, excluding sensitive credentials."""
        sensitive_keywords = ["token", "key", "secret", "password", "credential", "auth"]

        self.log("=== Environment Variables ===")
        env_vars = dict(os.environ)

        # Sort for consistent output
        for key in sorted(env_vars.keys()):
            value = env_vars[key]

            # Check if this is a sensitive variable
            is_sensitive = any(keyword in key.lower() for keyword in sensitive_keywords)

            if is_sensitive:
                # Show only first few characters for sensitive vars
                if len(value) > 8:
                    masked_value = f"{value[:4]}...{value[-4:]}"
                else:
                    masked_value = "***"
                self.log(f"  {key}={masked_value}")
            else:
                self.log(f"  {key}={value}")

        self.log("=== End Environment Variables ===")

    def _check_log_for_errors(self, line: str) -> tuple[str, bool] | None:
        """Check if a log line matches any error patterns."""
        return check_log_for_errors(
            line, self.compiled_fatal_patterns, self.compiled_retryable_patterns
        )

    def _process_log_line(self, line_str: str, process: subprocess.Popen):
        """Process a single log line: ship it and check for error patterns."""
        # Determine log level from output patterns
        level = "info"
        if "ERROR" in line_str or "Error" in line_str:
            level = "error"
        elif "WARNING" in line_str or "Warning" in line_str:
            level = "warning"

        # Ship to platform (via log() method which buffers and prints)
        self.log(line_str, level=level, source="lightx2v")

        # Check for error patterns
        error_match = self._check_log_for_errors(line_str)
        if error_match and not self.error_detected:
            matched_pattern, is_fatal = error_match
            self.error_detected = True
            self.error_is_fatal = is_fatal
            error_type = "FATAL" if is_fatal else "RETRYABLE"
            self.error_message = (
                f"{error_type} LightX2V error detected (pattern: {matched_pattern}): {line_str}"
            )
            self.log(f"ERROR DETECTED: {self.error_message}", level="error")

            # Send appropriate error heartbeat
            status = "fatal" if is_fatal else "error"
            self.send_heartbeat(status)

            # Trigger shutdown
            self.log(
                f"Initiating worker shutdown due to {error_type} LightX2V error",
                level="error",
            )
            self.running = False

            # Terminate engine process
            try:
                process.terminate()
            except Exception:
                pass

    def _monitor_engine_logs(self, process: subprocess.Popen):
        """
        Monitor LightX2V process logs in real-time and detect fatal errors.
        Runs in a separate thread.
        """
        self.log("Starting LightX2V log monitor thread")

        try:
            while self.running and process.poll() is None:
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        line_str = line.strip()
                        if line_str:
                            self._process_log_line(line_str, process)
                            if self.error_detected:
                                break

            # Drain remaining output after process exits
            if process.stdout:
                for line in process.stdout:
                    line_str = line.strip()
                    if line_str:
                        self._process_log_line(line_str, process)

            # If the process exited unexpectedly (not due to a matched error
            # pattern or a graceful shutdown), send an error heartbeat so the
            # scheduler can clean up.
            if not self.error_detected and self.running and process.poll() is not None:
                exit_code = process.returncode
                if exit_code is not None and exit_code != 0:
                    self.error_detected = True
                    self.error_is_fatal = False
                    self.error_message = f"LightX2V process exited with code {exit_code}"
                    self.log(self.error_message, level="error")
                    self.send_heartbeat("error")
                    self.running = False

        except Exception as e:
            self.log(f"Error in LightX2V log monitor: {e}", level="error")

        self.log("LightX2V log monitor thread exiting")

    def validate_config(self):
        ws_url = os.environ.get("CASOLA_WS_URL")
        if not ws_url:
            self.log("ERROR: CASOLA_WS_URL environment variable is required")
            return False
        return True

    def _start_engine_process(self) -> bool:
        """
        Start LightX2V server as a subprocess with log monitoring.
        Returns True if process started successfully, False otherwise.
        """
        try:
            # --model_path, --model_cls, --host, --port are formal argparse args.
            # --task and --config_json rely on LightX2V's dynamic arg passthrough
            # (parse_known_args + setattr) — they are not formally declared but are
            # required by the inference service at runtime.
            cmd = [
                "python3",
                "-m",
                "lightx2v.server",
                "--model_path",
                self.model_path,
                "--model_cls",
                self.model_cls,
                "--task",
                self.lightx2v_task,
                "--config_json",
                self.config_json,
                "--host",
                self.lightx2v_host,
                "--port",
                self.lightx2v_port,
            ]

            # Add extra args if provided
            extra_args = os.environ.get("LIGHTX2V_EXTRA_ARGS", "")
            if extra_args:
                cmd.extend(extra_args.split())

            self.engine_start_time = time.time()
            self.log(f"Starting LightX2V process: {' '.join(cmd)}")

            # Start LightX2V with stderr redirected to stdout for unified log monitoring
            self.engine_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )

            self.log(f"LightX2V process started (PID: {self.engine_process.pid})")

            # Start log monitoring thread immediately
            self.engine_log_monitor_thread = threading.Thread(
                target=self._monitor_engine_logs, args=(self.engine_process,), daemon=True
            )
            self.engine_log_monitor_thread.start()

            return True

        except Exception as e:
            self.log(f"Failed to start LightX2V process: {e}")
            return False

    def check_engine_health(self) -> bool:
        """Check if LightX2V /v1/service/status endpoint is reachable."""
        try:
            response = requests.get(f"{self.lightx2v_url}/v1/service/status", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_readiness(self) -> bool:
        """Wait for LightX2V /v1/service/status to return 200."""
        self.log(f"Waiting for LightX2V at {self.lightx2v_url}/v1/service/status...")
        start_time = time.time()

        while time.time() - start_time < self.lightx2v_startup_timeout:
            if self.error_detected:
                self.log("LightX2V error detected during startup, aborting readiness wait")
                return False
            if self.engine_process and self.engine_process.poll() is not None:
                self.log(
                    f"LightX2V process exited (code {self.engine_process.returncode}) during startup"
                )
                return False
            if not self.running:
                return False
            if self.check_engine_health():
                self.log("LightX2V health check passed (200 OK)")
                return True
            time.sleep(2)

        self.log(f"ERROR: LightX2V did not become healthy within {self.lightx2v_startup_timeout}s")
        return False

    def send_heartbeat(self, status: str = "active") -> bool:
        if not self.api_url or not self.api_token:
            return True  # Heartbeats are optional in WS mode
        try:
            payload = {"instance_id": self.instance_id, "queue_id": self.queue_id, "status": status}

            # Include error details for fatal errors
            if status == "fatal" and self.error_message:
                payload["error_message"] = self.error_message

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }

            self.log(f"Sending heartbeat: status={status}")
            response = requests.post(
                f"{self.api_url}/api/heartbeats", json=payload, headers=headers, timeout=10
            )

            if response.status_code >= 400:
                self.log(f"Heartbeat failed with status {response.status_code}: {response.text}")
                return False

            self.last_heartbeat = time.time()
            return True

        except Exception as e:
            self.log(f"Failed to send heartbeat: {e}")
            return False

    # Lease renewal is handled by the transport layer's auto-renewal loop,
    # which renews all active persisted jobs every 30s. This is sufficient
    # since the server-side lease is 300s. Short jobs complete before the
    # first renewal fires, avoiding unnecessary renewals.

    IMAGE_SIZES = {
        "square_hd": (1024, 1024),
        "square": (512, 512),
        "portrait_4_3": (768, 1024),
        "portrait_16_9": (768, 1344),
        "landscape_4_3": (1024, 768),
        "landscape_16_9": (1344, 768),
    }

    def _submit_text_to_video(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a text-to-video task to LightX2V and poll until complete."""
        prompt = payload.get("prompt", "")
        if not prompt:
            raise ValueError("No prompt provided in payload")

        width = payload.get("width", 1280)
        height = payload.get("height", 720)
        num_frames = payload.get("num_frames", 81)
        num_inference_steps = payload.get("num_inference_steps", 50)
        fps = payload.get("fps", 16)
        seed = payload.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        negative_prompt = payload.get("negative_prompt", "")

        self.log(
            f"Submitting T2V: prompt={prompt[:80]}..., {width}x{height}, "
            f"{num_frames} frames, {num_inference_steps} steps"
        )

        request_body = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "infer_steps": num_inference_steps,
            "seed": seed,
            "target_video_length": num_frames,
            "target_fps": fps,
            "target_shape": [num_frames, height, width],
        }

        start_time = time.time()
        resp = requests.post(
            f"{self.lightx2v_url}/v1/tasks/video",
            json=request_body,
            timeout=30,
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data["task_id"]

        self.log(f"LightX2V task submitted: {task_id}")

        # Poll for completion
        video_bytes = self._poll_task(task_id)
        inference_time = time.time() - start_time
        self.log(f"T2V completed: {len(video_bytes)} bytes in {inference_time:.1f}s")

        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "video_b64": video_b64,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "inference_time_seconds": inference_time,
            "seed_used": seed,
        }

    def _submit_image_to_video(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit an image-to-video task to LightX2V and poll until complete."""
        image_url = payload.get("image_url", "")
        if not image_url:
            raise ValueError("No image_url provided in payload")

        prompt = payload.get("prompt", "")
        width = payload.get("width", 1280)
        height = payload.get("height", 720)
        num_frames = payload.get("num_frames", 81)
        num_inference_steps = payload.get("num_inference_steps", 50)
        fps = payload.get("fps", 16)
        seed = payload.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        negative_prompt = payload.get("negative_prompt", "")

        self.log(f"Submitting I2V: image_url={image_url}, {width}x{height}, {num_frames} frames")

        # Download source image
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()

        # LightX2V's image_path field accepts base64-encoded strings or HTTP URLs
        img_b64 = base64.b64encode(img_response.content).decode("utf-8")

        request_body = {
            "prompt": prompt or "Animate this image naturally",
            "negative_prompt": negative_prompt,
            "image_path": img_b64,
            "infer_steps": num_inference_steps,
            "seed": seed,
            "target_video_length": num_frames,
            "target_fps": fps,
            "target_shape": [num_frames, height, width],
        }

        start_time = time.time()
        resp = requests.post(
            f"{self.lightx2v_url}/v1/tasks/video",
            json=request_body,
            timeout=30,
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data["task_id"]

        self.log(f"LightX2V I2V task submitted: {task_id}")

        # Poll for completion
        video_bytes = self._poll_task(task_id)
        inference_time = time.time() - start_time
        self.log(f"I2V completed: {len(video_bytes)} bytes in {inference_time:.1f}s")

        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "video_b64": video_b64,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "inference_time_seconds": inference_time,
            "seed_used": seed,
        }

    def _poll_task(self, task_id: str) -> bytes:
        """Poll LightX2V task status until COMPLETED, then download the result."""
        poll_interval = 2
        task_timeout = 900  # 15 minutes max for video generation

        start_time = time.time()
        while time.time() - start_time < task_timeout:
            if not self.running:
                raise RuntimeError("Worker shutting down during job processing")

            try:
                status_resp = requests.get(
                    f"{self.lightx2v_url}/v1/tasks/{task_id}/status",
                    timeout=10,
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    task_status = status_data.get("status", "").upper()

                    if task_status == "COMPLETED":
                        # Download result
                        result_resp = requests.get(
                            f"{self.lightx2v_url}/v1/tasks/{task_id}/result",
                            timeout=120,
                        )
                        result_resp.raise_for_status()
                        return result_resp.content

                    elif task_status == "FAILED":
                        error_msg = status_data.get("error", "Unknown LightX2V error")
                        raise RuntimeError(f"LightX2V task failed: {error_msg}")

                    elif task_status == "CANCELLED":
                        raise RuntimeError("LightX2V task was cancelled")

            except requests.exceptions.RequestException as e:
                # Transient network error, keep polling
                self.log(f"Poll error (will retry): {e}", level="warning")

            time.sleep(poll_interval)

        raise RuntimeError(f"LightX2V task {task_id} timed out after {task_timeout}s")

    def _complete_video_job(
        self,
        job_id: str,
        config_id: str,
        task: str,
        video_result: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Build result payload from video_result, upload to R2 if available, and complete the job."""
        result_data = {
            "status": "success",
            "config_id": config_id,
            "task": task,
            "width": video_result["width"],
            "height": video_result["height"],
            "num_frames": video_result["num_frames"],
            "fps": video_result["fps"],
            "duration_seconds": video_result["duration_seconds"],
            "inference_time_seconds": video_result["inference_time_seconds"],
            "seed_used": video_result.get("seed_used"),
            "processed_at": datetime.now().isoformat(),
        }
        if extra:
            result_data.update(extra)

        if self.transport.has_media_upload_urls(job_id):
            video_bytes = base64.b64decode(video_result["video_b64"])
            self.transport.upload_to_r2(job_id, 0, video_bytes, "video/mp4")
            result_data["video_content_type"] = "video/mp4"
            result_data["media_count"] = 1
        else:
            result_data["video_b64"] = video_result["video_b64"]

        return self.transport.complete_job(job_id, result_data)

    def _submit_text_to_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a text-to-image task to LightX2V and poll until complete."""
        prompt = payload.get("prompt", "")
        if not prompt:
            raise ValueError("No prompt provided in payload")

        negative_prompt = payload.get("negative_prompt", "")
        image_size = payload.get("image_size", "square_hd")
        width, height = self.IMAGE_SIZES.get(image_size, self.IMAGE_SIZES["square_hd"])
        num_inference_steps = payload.get("num_inference_steps", 28)
        guidance_scale = payload.get("guidance_scale", 5.0)
        num_images = payload.get("num_images", 1)
        seed = payload.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        self.log(
            f"Submitting T2I: prompt={prompt[:80]}..., {width}x{height}, "
            f"{num_inference_steps} steps, {num_images} image(s)"
        )

        images_bytes: list[bytes] = []
        start_time = time.time()

        for i in range(num_images):
            request_body = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "infer_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed + i,
                "target_shape": [height, width],
            }
            loras = payload.get("loras")
            if loras:
                request_body["loras"] = loras

            resp = requests.post(
                f"{self.lightx2v_url}/v1/tasks/image",
                json=request_body,
                timeout=30,
            )
            resp.raise_for_status()
            task_id = resp.json()["task_id"]
            self.log(f"LightX2V T2I task submitted: {task_id} ({i + 1}/{num_images})")

            image_bytes = self._poll_task(task_id)
            images_bytes.append(image_bytes)

        inference_time = time.time() - start_time
        self.log(f"T2I completed: {len(images_bytes)} image(s) in {inference_time:.1f}s")

        return {
            "images": images_bytes,
            "width": width,
            "height": height,
            "inference_time_seconds": inference_time,
            "seed_used": seed,
        }

    def _submit_image_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit an image-edit task to LightX2V and poll until complete."""
        image_url = payload.get("image_url", "")
        if not image_url:
            raise ValueError("No image_url provided in payload")

        prompt = payload.get("prompt", "")
        mask_url = payload.get("mask_url")
        strength = payload.get("strength")
        num_inference_steps = payload.get("num_inference_steps", 28)
        guidance_scale = payload.get("guidance_scale", 5.0)
        num_images = payload.get("num_images", 1)
        seed = payload.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        self.log(f"Submitting image-edit: image_url={image_url}, prompt={prompt[:80]}...")

        # Download source image and base64-encode
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        img_b64 = base64.b64encode(img_response.content).decode("utf-8")

        images_bytes: list[bytes] = []
        start_time = time.time()

        for i in range(num_images):
            request_body = {
                "prompt": prompt,
                "image_path": img_b64,
                "infer_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed + i,
            }
            if mask_url:
                mask_response = requests.get(mask_url, timeout=30)
                mask_response.raise_for_status()
                request_body["mask_path"] = base64.b64encode(mask_response.content).decode("utf-8")
            if strength is not None:
                request_body["strength"] = strength
            loras = payload.get("loras")
            if loras:
                request_body["loras"] = loras

            resp = requests.post(
                f"{self.lightx2v_url}/v1/tasks/image",
                json=request_body,
                timeout=30,
            )
            resp.raise_for_status()
            task_id = resp.json()["task_id"]
            self.log(f"LightX2V image-edit task submitted: {task_id} ({i + 1}/{num_images})")

            image_bytes = self._poll_task(task_id)
            images_bytes.append(image_bytes)

        inference_time = time.time() - start_time
        self.log(f"Image-edit completed: {len(images_bytes)} image(s) in {inference_time:.1f}s")

        # Get dimensions from source image (best-effort)
        try:
            import io

            from PIL import Image

            src_img = Image.open(io.BytesIO(img_response.content))
            width, height = src_img.size
        except Exception:
            width, height = 0, 0

        return {
            "images": images_bytes,
            "width": width,
            "height": height,
            "inference_time_seconds": inference_time,
            "seed_used": seed,
        }

    def _complete_image_job(
        self,
        job_id: str,
        config_id: str,
        task: str,
        image_result: dict[str, Any],
    ) -> bool:
        """Build result payload from image_result, upload to R2 if available, and complete the job."""
        width = image_result["width"]
        height = image_result["height"]
        use_r2 = self.transport.has_media_upload_urls(job_id)

        images = []
        for i, img_bytes in enumerate(image_result["images"]):
            if use_r2:
                self.transport.upload_to_r2(job_id, i, img_bytes, "image/png")
                images.append(
                    {
                        "media_index": i,
                        "width": width,
                        "height": height,
                        "content_type": "image/png",
                    }
                )
            else:
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                images.append(
                    {
                        "url": f"data:image/png;base64,{img_b64}",
                        "width": width,
                        "height": height,
                        "content_type": "image/png",
                    }
                )

        result_data = {
            "images": images,
            "seed": image_result["seed_used"],
            "timings": {"inference": round(image_result["inference_time_seconds"], 2)},
        }
        if use_r2:
            result_data["media_count"] = len(images)

        return self.transport.complete_job(job_id, result_data)

    def process_job(self, job: dict[str, Any]) -> bool:
        job_id = job.get("id")
        config_id = job.get("config_id")
        payload = job.get("payload", {})
        task = payload.get("task", "")

        with self.job_lock:
            self.current_job_id = job_id
        if self.metrics_reporter:
            self.metrics_reporter.set_current_job(job_id)

        self.log(
            f"Processing job {job_id} (config_id: {config_id}, task: {task})",
            job_id=job_id,
            config_id=config_id,
        )

        try:
            if task == "fal/text-to-video":
                video_result = self._submit_text_to_video(payload)
                result = self._complete_video_job(job_id, config_id, task, video_result)

            elif task == "fal/image-to-video":
                video_result = self._submit_image_to_video(payload)
                result = self._complete_video_job(
                    job_id,
                    config_id,
                    task,
                    video_result,
                    extra={"source_image_url": payload.get("image_url", "")},
                )

            elif task == "fal/text-to-image":
                image_result = self._submit_text_to_image(payload)
                result = self._complete_image_job(job_id, config_id, task, image_result)

            elif task == "fal/image-edit":
                image_result = self._submit_image_edit(payload)
                result = self._complete_image_job(job_id, config_id, task, image_result)

            else:
                raise ValueError(f"Unsupported task: {task!r} (config_id={config_id})")

            with self.job_lock:
                self.current_job_id = None
            if self.metrics_reporter:
                self.metrics_reporter.set_current_job(None)

            return result

        except Exception as e:
            self.log(
                f"Error processing job {job_id}: {e}",
                level="error",
                job_id=job_id,
                config_id=config_id,
            )
            result = self.transport.fail_job(job_id, str(e))

            with self.job_lock:
                self.current_job_id = None
            if self.metrics_reporter:
                self.metrics_reporter.set_current_job(None)

            return result

    def _on_first_active(self):
        """Send first 'active' heartbeat and log startup timing breakdown."""
        self.send_heartbeat("active")
        now = time.time()
        if self.scheduler_created_at:
            self.log(f"Time since scheduler creation: {now - self.scheduler_created_at:.1f}s")
        else:
            self.log("Time since scheduler creation: unknown (CASOLA_CREATED_AT not set)")
        self.log(f"Time since container startup: {now - self.container_start_time:.1f}s")
        if self.engine_start_time:
            self.log(f"Time since LightX2V process start: {now - self.engine_start_time:.1f}s")

    def _heartbeat_loop(self):
        while self.running:
            try:
                # Sleep first, as the initial heartbeat was sent in run()
                time.sleep(self.heartbeat_interval)

                if self.running:
                    # Enforce health check before sending heartbeat
                    if self.check_engine_health():
                        self.send_heartbeat("active")
                    else:
                        self.log("Skipping heartbeat: LightX2V health check failed")
            except Exception as e:
                self.log(f"Error in heartbeat loop: {e}")

    def _collect_engine_metrics(self) -> dict[str, Any]:
        """Collect LightX2V engine metrics for the system metrics reporter."""
        metrics: dict[str, Any] = {}
        try:
            response = requests.get(f"{self.lightx2v_url}/v1/tasks/queue/status", timeout=2)
            if response.status_code == 200:
                data = response.json()
                pending = data.get("pending_count", 0)
                active = data.get("active_count", 0)
                metrics["lightx2v_queue_depth"] = pending + active
        except Exception:
            pass
        return metrics

    def _transport_loop(self):
        """Run the transport in a thread (blocking call)."""
        try:
            self.transport.start()
        except Exception:
            self.log("Transport loop exited with error")

    def run(self):
        if not self.validate_config():
            sys.exit(1)

        self.log("Starting LightX2V worker...")
        self.log(f"Instance ID: {self.instance_id}")
        self.log(f"Worker ID: {self.worker_id}")
        self.log(f"Model: {self.model_cls} ({self.model_path})")
        self.log(f"Task: {self.lightx2v_task}")
        if self.api_url:
            self.log(f"API URL: {self.api_url}")

        # Print environment variables for debugging
        self.print_environment_variables()

        # 1. Ship startup diagnostics (GPU/CUDA/model cache) into log buffer
        from casola_worker.gpu_health import run_gpu_health_checks, run_startup_diagnostics

        for line in run_startup_diagnostics():
            self.log(line, source="diagnostics")

        # 2. Send "starting" heartbeat immediately
        self.send_heartbeat("starting")

        # 3. Run pre-startup GPU health checks
        health_result = run_gpu_health_checks(
            expected_gpu_name=os.environ.get("CASOLA_EXPECTED_GPU_NAME"),
            expected_vram_gb=float(v) if (v := os.environ.get("CASOLA_EXPECTED_VRAM_GB")) else None,
        )
        for detail in health_result.details:
            self.log(detail, source="diagnostics")
        if not health_result.passed:
            self.error_message = health_result.error_message
            self.error_detected = True
            self.error_is_fatal = True
            self.log(f"GPU health check failed: {health_result.error_message}")
            self.send_heartbeat("fatal")
            sys.exit(1)
        self.log(f"GPU health checks passed ({health_result.checks_run} checks)")

        # 4. Start LightX2V process with log monitoring
        if not self._start_engine_process():
            self.log("Failed to start LightX2V process")
            self.send_heartbeat("error")
            sys.exit(1)

        # 5. Wait for LightX2V to be healthy
        if not self.wait_for_readiness():
            self.log("Failed to start: LightX2V did not become healthy.")
            # If an error was detected during startup, use appropriate status
            if self.error_detected:
                status = "fatal" if self.error_is_fatal else "error"
                self.send_heartbeat(status)
            else:
                self.send_heartbeat("failed")
            # Clean up engine process
            if self.engine_process:
                self.engine_process.terminate()
                self.engine_process.wait(timeout=5)
            sys.exit(1)

        # 6. Build transport config from env vars
        config = build_transport_config(
            worker_id=self.worker_id,
            config_id=self.config_id,
            max_jobs=self.max_jobs,
            lease_seconds=self.lease_seconds,
            api_token=self.api_token,
        )

        self.transport = create_transport(config)
        self.transport.on_job = self.process_job
        self.transport.on_connected = lambda: self._on_first_active()

        # 6. Start heartbeat thread (optional — needs API URL/token)
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        if self.api_url and self.api_token:
            heartbeat_thread.start()

        # 6b. Start log flush thread
        log_flush_thread = threading.Thread(target=self._log_flush_loop, daemon=True)
        log_flush_thread.start()

        # 6c. Start system metrics reporter
        self.metrics_reporter = SystemMetricsReporter(
            instance_id=self.instance_id,
            queue_id=self.queue_id,
            config_id=self.config_id,
            api_token=self.api_token or "",
            engine_metrics_fn=self._collect_engine_metrics,
            log_fn=self.log,
        )
        self.metrics_reporter.start()

        # 7. Start transport in a thread so main loop can monitor engine health
        transport_thread = threading.Thread(target=self._transport_loop, daemon=True)
        transport_thread.start()

        try:
            while self.running:
                # Check if LightX2V process died unexpectedly
                if self.engine_process and self.engine_process.poll() is not None:
                    if not self.error_detected:
                        self.log("LightX2V process died unexpectedly")
                        self.send_heartbeat("error")
                        self.running = False
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            self.log("Received keyboard interrupt")
            self.running = False

        # Stop the transport
        if self.transport:
            self.transport.stop()

        # Flush and stop metrics reporter
        if self.metrics_reporter:
            self.metrics_reporter.flush()
            self.metrics_reporter.stop()

        if self.api_url and self.api_token:
            heartbeat_thread.join(timeout=5)
        transport_thread.join(timeout=5)

        # Clean up LightX2V process
        if self.engine_process:
            self.log("Terminating LightX2V process")
            try:
                self.engine_process.terminate()
                self.engine_process.wait(timeout=5)
            except Exception:
                self.log("Force killing LightX2V process")
                self.engine_process.kill()

        # Exit with error code if error was detected
        if self.error_detected:
            sys.exit(1)


if __name__ == "__main__":
    worker = LightX2VWorker()
    worker.run()
