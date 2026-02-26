#!/usr/bin/env python3

import base64
import copy
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

# Allow importing casola_worker from parent directory (local dev)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from casola_worker.system_metrics import SystemMetricsReporter
from casola_worker.transport import QueueTransport, create_transport


class ComfyUIWorker:
    # Regex patterns that indicate FATAL errors (permanent, host should be marked as bad)
    # NOTE: Avoid broad patterns like "libcuda\.so", "cannot find -lcuda",
    # "CUDA.*linker", "ld returned 1 exit status" — these false-positive on
    # benign Triton JIT linker warnings during torch.compile.
    FATAL_ERROR_PATTERNS = [
        r"CUDA driver version is insufficient",
        r"CUDA version.*incompatible",
        r"cannot open shared object file.*libcuda",
        r"Incompatible CUDA version",
    ]

    # Regex patterns that indicate retryable errors (temporary, can retry on same or different host)
    RETRYABLE_ERROR_PATTERNS = [
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"torch\.cuda\.OutOfMemoryError",
        r"Cannot allocate memory",
        r"GPU memory allocation failed",
        r"Error downloading model",
        r"Failed to download",
        r"RuntimeError.*CUDA",
        r"Segmentation fault",
        r"core dumped",
        r"NCCL error",
    ]

    def __init__(self):
        self.api_url = os.environ.get("CASOLA_API_URL")
        self.api_token = os.environ.get("CASOLA_API_TOKEN")

        container_id = os.environ.get("CONTAINER_ID", str(os.getpid()))
        self.instance_id = os.environ.get("CASOLA_INSTANCE_ID", f"vast-{container_id}")
        self.queue_id = os.environ.get("CASOLA_QUEUE_ID", "comfy-queue")
        self.worker_id = self.instance_id

        self.heartbeat_interval = int(os.environ.get("CASOLA_HEARTBEAT_INTERVAL", "60"))
        self.max_jobs = int(os.environ.get("CASOLA_MAX_JOBS", "1"))
        self.lease_seconds = int(os.environ.get("CASOLA_LEASE_SECONDS", "60"))
        self.shutdown_grace_period = float(os.environ.get("CASOLA_SHUTDOWN_GRACE_PERIOD", "4.0"))

        # Read config_id from environment (set by scheduler)
        self.config_id = os.environ.get("CASOLA_CONFIG_ID", "")
        if not self.config_id:
            raise ValueError("CASOLA_CONFIG_ID environment variable is required")

        self.comfyui_host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
        self.comfyui_port = os.environ.get("COMFYUI_PORT", "8188")
        self.comfyui_url = f"http://{self.comfyui_host}:{self.comfyui_port}"
        self.comfyui_startup_timeout = int(os.environ.get("COMFYUI_STARTUP_TIMEOUT", "600"))

        self.running = True
        self.last_heartbeat = 0
        self.current_job_id = None
        self.job_lock = threading.Lock()

        self.transport: Optional[QueueTransport] = None

        self.comfyui_process = None
        self.comfyui_log_monitor_thread = None
        self.error_detected = False
        self.error_message = None
        self.error_is_fatal = False

        # Loaded from /app/config.json and /app/workflow.json
        self.config = None
        self.workflow_template = None

        # Compile regex patterns for efficiency
        self.compiled_fatal_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.FATAL_ERROR_PATTERNS
        ]
        self.compiled_retryable_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.RETRYABLE_ERROR_PATTERNS
        ]

        # Log shipping
        self.log_buffer: List[Dict[str, Any]] = []
        self.log_lock_logs = threading.Lock()
        self.last_log_flush = time.time()
        self.log_flush_interval = int(os.environ.get("CASOLA_LOG_FLUSH_INTERVAL", "30"))
        self.ws_url = os.environ.get("CASOLA_WS_URL", "")

        # System metrics reporter (initialized in run() after engine is ready)
        self.metrics_reporter: Optional[SystemMetricsReporter] = None

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
        job_id: Optional[str] = None,
        config_id: Optional[str] = None,
    ):
        """Log a message from the worker or subprocess (ComfyUI)."""
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

        with self.log_lock_logs:
            self.log_buffer.append(entry)

            # Prevent unbounded memory growth
            if len(self.log_buffer) > 10000:
                self.log_buffer = self.log_buffer[-5000:]

    def flush_logs(self):
        """Send buffered logs to Queue DO via HTTP."""
        with self.log_lock_logs:
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
            with self.log_lock_logs:
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

    def _check_log_for_errors(self, line: str) -> Optional[tuple]:
        """
        Check if a log line matches any error patterns.
        Returns tuple of (matched_pattern, is_fatal) if found, None otherwise.
        """
        # Check fatal patterns first
        for pattern in self.compiled_fatal_patterns:
            if pattern.search(line):
                return (pattern.pattern, True)

        # Check retryable patterns
        for pattern in self.compiled_retryable_patterns:
            if pattern.search(line):
                return (pattern.pattern, False)

        return None

    def _monitor_comfyui_logs(self, process: subprocess.Popen):
        """
        Monitor ComfyUI process logs in real-time and detect fatal errors.
        Runs in a separate thread.
        """
        self.log("Starting ComfyUI log monitor thread")

        try:
            while self.running and process.poll() is None:
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        line_str = line.strip()
                        if line_str:
                            # Determine log level from ComfyUI output patterns
                            level = "info"
                            if "ERROR" in line_str or "Error" in line_str:
                                level = "error"
                            elif "WARNING" in line_str or "Warning" in line_str:
                                level = "warning"

                            # Ship to platform (via log() method which buffers and prints)
                            self.log(line_str, level=level, source="comfyui")

                            # Check for error patterns
                            error_match = self._check_log_for_errors(line_str)
                            if error_match and not self.error_detected:
                                matched_pattern, is_fatal = error_match
                                self.error_detected = True
                                self.error_is_fatal = is_fatal
                                error_type = "FATAL" if is_fatal else "RETRYABLE"
                                self.error_message = f"{error_type} ComfyUI error detected (pattern: {matched_pattern}): {line_str}"
                                self.log(f"ERROR DETECTED: {self.error_message}", level="error")

                                # Send appropriate error heartbeat
                                status = "fatal" if is_fatal else "error"
                                self.send_heartbeat(status)

                                # Trigger shutdown
                                self.log(
                                    f"Initiating worker shutdown due to {error_type} ComfyUI error",
                                    level="error",
                                )
                                self.running = False

                                # Terminate ComfyUI process
                                try:
                                    process.terminate()
                                except Exception:
                                    pass

                                break
        except Exception as e:
            self.log(f"Error in ComfyUI log monitor: {e}", level="error")

        self.log("ComfyUI log monitor thread exiting")

    def validate_config(self):
        ws_url = os.environ.get("CASOLA_WS_URL")
        if not ws_url:
            self.log("ERROR: CASOLA_WS_URL environment variable is required")
            return False
        return True

    def load_workflow_config(self) -> bool:
        """Load config.json and workflow.json from /app/."""
        try:
            with open("/app/config.json", "r") as f:
                self.config = json.load(f)
            self.log(
                f"Loaded config.json: {json.dumps({k: v for k, v in self.config.items() if k != 'workflow_params'}, indent=2)}"
            )
        except FileNotFoundError:
            self.log("ERROR: /app/config.json not found")
            return False
        except json.JSONDecodeError as e:
            self.log(f"ERROR: Invalid JSON in /app/config.json: {e}")
            return False

        try:
            with open("/app/workflow.json", "r") as f:
                self.workflow_template = json.load(f)
            self.log(f"Loaded workflow.json with {len(self.workflow_template)} nodes")
        except FileNotFoundError:
            self.log("ERROR: /app/workflow.json not found")
            return False
        except json.JSONDecodeError as e:
            self.log(f"ERROR: Invalid JSON in /app/workflow.json: {e}")
            return False

        # Validate required config fields
        required_fields = ["image_sizes", "defaults", "workflow_params"]
        for field in required_fields:
            if field not in self.config:
                self.log(f"ERROR: Missing required field '{field}' in config.json")
                return False

        return True

    def _start_comfyui_process(self) -> bool:
        """
        Start ComfyUI server as a subprocess with log monitoring.
        Returns True if process started successfully, False otherwise.
        """
        try:
            cmd = [
                "python3",
                "/app/comfyui/main.py",
                "--listen",
                self.comfyui_host,
                "--port",
                self.comfyui_port,
            ]

            # Add extra args from config
            if self.config and "comfyui_args" in self.config:
                cmd.extend(self.config["comfyui_args"])

            self.log(f"Starting ComfyUI process: {' '.join(cmd)}")

            # Start ComfyUI with stderr redirected to stdout for unified log monitoring
            self.comfyui_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )

            self.log(f"ComfyUI process started (PID: {self.comfyui_process.pid})")

            # Start log monitoring thread immediately
            self.comfyui_log_monitor_thread = threading.Thread(
                target=self._monitor_comfyui_logs, args=(self.comfyui_process,), daemon=True
            )
            self.comfyui_log_monitor_thread.start()

            return True

        except Exception as e:
            self.log(f"Failed to start ComfyUI process: {e}")
            return False

    def check_comfyui_health(self) -> bool:
        """Checks if ComfyUI /system_stats endpoint is reachable."""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_comfyui_readiness(self) -> bool:
        """Waits for ComfyUI /system_stats to return 200."""
        self.log(f"Waiting for ComfyUI at {self.comfyui_url}/system_stats...")
        start_time = time.time()

        while time.time() - start_time < self.comfyui_startup_timeout:
            if not self.running:
                return False
            if self.error_detected:
                return False
            if self.check_comfyui_health():
                self.log("ComfyUI health check passed (200 OK)")
                return True
            time.sleep(2)

        self.log(f"ERROR: ComfyUI did not become healthy within {self.comfyui_startup_timeout}s")
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

    def _set_workflow_param(self, workflow: dict, node_id: str, field_path: List[str], value: Any):
        """Set a value in the workflow at the given node and field path."""
        if node_id not in workflow:
            self.log(f"WARNING: Node '{node_id}' not found in workflow, skipping")
            return

        obj = workflow[node_id]
        for key in field_path[:-1]:
            obj = obj[key]
        obj[field_path[-1]] = value

    def process_job(self, job: Dict[str, Any]) -> bool:
        job_id = job.get("id")
        config_id = job.get("config_id")
        payload = job.get("payload", {})

        with self.job_lock:
            self.current_job_id = job_id
        if self.metrics_reporter:
            self.metrics_reporter.set_current_job(job_id)

        self.log(f"Processing job {job_id} (config_id: {config_id})")

        try:
            start_time = time.time()

            # Deep-copy workflow template
            workflow = copy.deepcopy(self.workflow_template)
            params = self.config["workflow_params"]
            defaults = self.config["defaults"]

            # Patch positive prompt
            prompt_text = payload.get("prompt", "")
            if not prompt_text:
                raise ValueError("No prompt provided in payload")
            self._set_workflow_param(
                workflow,
                params["positive_prompt_node"],
                params["positive_prompt_field"],
                prompt_text,
            )

            # Patch negative prompt
            negative_prompt = payload.get("negative_prompt", "")
            self._set_workflow_param(
                workflow,
                params["negative_prompt_node"],
                params["negative_prompt_field"],
                negative_prompt,
            )

            # Patch image size (width/height)
            image_size_name = payload.get("image_size", defaults.get("image_size", "square_hd"))
            image_sizes = self.config["image_sizes"]
            if image_size_name in image_sizes:
                width, height = image_sizes[image_size_name]
            else:
                self.log(f"Unknown image_size '{image_size_name}', using square_hd")
                width, height = image_sizes.get("square_hd", [1024, 1024])

            self._set_workflow_param(workflow, params["width_node"], params["width_field"], width)
            self._set_workflow_param(
                workflow, params["height_node"], params["height_field"], height
            )

            # Patch batch size (num_images)
            num_images = payload.get("num_images", defaults.get("num_images", 1))
            self._set_workflow_param(
                workflow, params["batch_size_node"], params["batch_size_field"], num_images
            )

            # Patch steps
            steps = payload.get("num_inference_steps", defaults.get("num_inference_steps", 28))
            self._set_workflow_param(workflow, params["steps_node"], params["steps_field"], steps)

            # Patch cfg (guidance_scale)
            cfg = payload.get("guidance_scale", defaults.get("guidance_scale", 5.0))
            self._set_workflow_param(workflow, params["cfg_node"], params["cfg_field"], cfg)

            # Patch seed
            seed = payload.get("seed")
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
            self._set_workflow_param(workflow, params["seed_node"], params["seed_field"], seed)

            self.log(
                f"Submitting workflow: size={image_size_name} ({width}x{height}), steps={steps}, cfg={cfg}, seed={seed}, batch={num_images}"
            )

            # Submit prompt to ComfyUI
            submit_response = requests.post(
                f"{self.comfyui_url}/prompt",
                json={"prompt": workflow, "client_id": "casola-worker"},
                timeout=30,
            )

            if submit_response.status_code != 200:
                raise RuntimeError(
                    f"ComfyUI prompt submission failed: {submit_response.status_code} - {submit_response.text}"
                )

            prompt_result = submit_response.json()
            prompt_id = prompt_result.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"No prompt_id in ComfyUI response: {prompt_result}")

            self.log(f"ComfyUI prompt submitted: {prompt_id}")

            # Poll /history/{prompt_id} until outputs appear
            poll_timeout = self.lease_seconds * 3  # generous timeout
            poll_start = time.time()

            while time.time() - poll_start < poll_timeout:
                if not self.running:
                    raise RuntimeError("Worker shutting down during job processing")

                try:
                    history_response = requests.get(
                        f"{self.comfyui_url}/history/{prompt_id}", timeout=5
                    )

                    if history_response.status_code == 200:
                        history = history_response.json()
                        if prompt_id in history:
                            prompt_history = history[prompt_id]

                            # Check for execution errors
                            status_data = prompt_history.get("status", {})
                            if status_data.get("status_str") == "error":
                                messages = status_data.get("messages", [])
                                error_msg = (
                                    str(messages) if messages else "Unknown ComfyUI execution error"
                                )
                                raise RuntimeError(f"ComfyUI execution error: {error_msg}")

                            outputs = prompt_history.get("outputs", {})
                            if outputs:
                                self.log(f"ComfyUI execution complete for {prompt_id}")
                                break
                except requests.exceptions.RequestException:
                    pass  # Transient network error, keep polling

                time.sleep(1)
            else:
                raise RuntimeError(f"ComfyUI execution timed out after {poll_timeout}s")

            # Collect output images
            images = []
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img_info in node_output["images"]:
                        filename = img_info.get("filename", "")
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")

                        # Download image from ComfyUI
                        view_params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": img_type,
                        }
                        img_response = requests.get(
                            f"{self.comfyui_url}/view", params=view_params, timeout=30
                        )

                        if img_response.status_code == 200:
                            img_base64 = base64.b64encode(img_response.content).decode("utf-8")
                            content_type = img_response.headers.get("Content-Type", "image/png")
                            images.append(
                                {
                                    "url": f"data:{content_type};base64,{img_base64}",
                                    "width": width,
                                    "height": height,
                                    "content_type": content_type,
                                }
                            )
                        else:
                            self.log(
                                f"Failed to download image {filename}: {img_response.status_code}"
                            )

            if not images:
                raise RuntimeError("ComfyUI produced no output images")

            elapsed = time.time() - start_time
            self.log(f"Generated {len(images)} image(s) in {elapsed:.1f}s")

            result_data = {
                "images": images,
                "seed": seed,
                "timings": {"inference": round(elapsed, 2)},
            }

            result = self.transport.complete_job(job_id, result_data)

            with self.job_lock:
                self.current_job_id = None
            if self.metrics_reporter:
                self.metrics_reporter.set_current_job(None)

            return result

        except Exception as e:
            self.log(f"Error processing job {job_id}: {e}")
            result = self.transport.fail_job(job_id, str(e))

            with self.job_lock:
                self.current_job_id = None
            if self.metrics_reporter:
                self.metrics_reporter.set_current_job(None)

            return result

    def _heartbeat_loop(self):
        # We start this thread AFTER the initial checks in run(), so we just loop.
        while self.running:
            try:
                # Sleep first, as the initial "active" heartbeat was sent in run()
                time.sleep(self.heartbeat_interval)

                if self.running:
                    # Enforce health check before sending heartbeat
                    if self.check_comfyui_health():
                        self.send_heartbeat("active")
                    else:
                        self.log("Skipping heartbeat: ComfyUI health check failed")
            except Exception as e:
                self.log(f"Error in heartbeat loop: {e}")

    def _collect_comfyui_metrics(self) -> Dict[str, Any]:
        """Collect ComfyUI engine metrics for the system metrics reporter."""
        metrics: Dict[str, Any] = {}
        try:
            response = requests.get(f"{self.comfyui_url}/queue", timeout=2)
            if response.status_code == 200:
                data = response.json()
                running = len(data.get("queue_running", []))
                pending = len(data.get("queue_pending", []))
                metrics["comfyui_queue_depth"] = running + pending
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

        self.log("Starting ComfyUI worker...")
        self.log(f"Instance ID: {self.instance_id}")
        self.log(f"Worker ID: {self.worker_id}")
        if self.api_url:
            self.log(f"API URL: {self.api_url}")

        # Print environment variables for debugging
        self.print_environment_variables()

        # 1. Load workflow config and template
        if not self.load_workflow_config():
            self.log("Failed to load workflow configuration")
            sys.exit(1)

        # 2. Send "starting" heartbeat immediately
        self.send_heartbeat("starting")

        # 3. Run pre-startup GPU health checks
        from casola_worker.gpu_health import run_gpu_health_checks

        health_result = run_gpu_health_checks(
            expected_gpu_name=os.environ.get("CASOLA_EXPECTED_GPU_NAME"),
            expected_vram_gb=float(v) if (v := os.environ.get("CASOLA_EXPECTED_VRAM_GB")) else None,
        )
        if not health_result.passed:
            self.error_message = health_result.error_message
            self.error_detected = True
            self.error_is_fatal = True
            self.log(f"GPU health check failed: {health_result.error_message}")
            self.send_heartbeat("fatal")
            sys.exit(1)
        self.log(f"GPU health checks passed ({health_result.checks_run} checks)")

        # 4. Start ComfyUI process with log monitoring
        if not self._start_comfyui_process():
            self.log("Failed to start ComfyUI process")
            self.send_heartbeat("error")
            sys.exit(1)

        # 5. Wait for ComfyUI to be healthy
        if not self.wait_for_comfyui_readiness():
            self.log("Failed to start: ComfyUI did not become healthy.")
            # If an error was detected during startup, use appropriate status
            if self.error_detected:
                status = "fatal" if self.error_is_fatal else "error"
                self.send_heartbeat(status)
            else:
                self.send_heartbeat("failed")
            # Clean up ComfyUI process
            if self.comfyui_process:
                self.comfyui_process.terminate()
                self.comfyui_process.wait(timeout=5)
            sys.exit(1)

        # 6. Build transport config from env vars
        config = {
            "worker_id": self.worker_id,
            "max_jobs": self.max_jobs,
            "lease_seconds": self.lease_seconds,
            "config_id": self.config_id,
            "ws_url": os.environ.get("CASOLA_WS_URL", ""),
            "capacity": int(os.environ.get("CASOLA_CAPACITY", "1")),
        }

        self.transport = create_transport(config)
        self.transport.on_job = self.process_job
        self.transport.on_connected = lambda: self.send_heartbeat("active")

        # 7. Start heartbeat thread (optional — needs API URL/token)
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        if self.api_url and self.api_token:
            heartbeat_thread.start()

        # 7b. Start log flush thread
        log_flush_thread = threading.Thread(target=self._log_flush_loop, daemon=True)
        log_flush_thread.start()

        # 7c. Start system metrics reporter
        self.metrics_reporter = SystemMetricsReporter(
            instance_id=self.instance_id,
            queue_id=self.queue_id,
            config_id=self.config_id,
            api_token=self.api_token or "",
            engine_metrics_fn=self._collect_comfyui_metrics,
            log_fn=self.log,
        )
        self.metrics_reporter.start()

        # 8. Start transport in a thread so main loop can monitor ComfyUI health
        transport_thread = threading.Thread(target=self._transport_loop, daemon=True)
        transport_thread.start()

        try:
            while self.running:
                # Check if ComfyUI process died unexpectedly
                if self.comfyui_process and self.comfyui_process.poll() is not None:
                    if not self.error_detected:
                        self.log("ComfyUI process died unexpectedly")
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

        # Clean up ComfyUI process
        if self.comfyui_process:
            self.log("Terminating ComfyUI process")
            try:
                self.comfyui_process.terminate()
                self.comfyui_process.wait(timeout=5)
            except Exception:
                self.log("Force killing ComfyUI process")
                self.comfyui_process.kill()

        # Exit with error code if error was detected
        if self.error_detected:
            sys.exit(1)


if __name__ == "__main__":
    worker = ComfyUIWorker()
    worker.run()
