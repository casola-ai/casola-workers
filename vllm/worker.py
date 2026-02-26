#!/usr/bin/env python3

import base64
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# Allow importing casola_worker from parent directory (local dev)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from casola_worker.system_metrics import SystemMetricsReporter
from casola_worker.transport import QueueTransport, create_transport


class VLLMWorker:
    # Regex patterns that indicate FATAL errors (permanent, host should be marked as bad)
    # NOTE: Avoid broad patterns like "libcuda\.so", "cannot find -lcuda",
    # "CUDA.*linker", "ld returned 1 exit status" — these false-positive on
    # benign Triton JIT linker warnings during torch.compile.
    FATAL_ERROR_PATTERNS = [
        r"CUDA driver version is insufficient",
        r"CUDA version.*incompatible",
        r"cannot open shared object file.*libcuda",
        r"Incompatible CUDA version",
        r"forward compatibility was attempted on non supported HW",
        r"unsupported display driver / cuda driver combination",
    ]

    # Regex patterns that indicate retryable errors (temporary, can retry on same or different host)
    RETRYABLE_ERROR_PATTERNS = [
        r"CUDA out of memory",
        r"RuntimeError.*CUDA",
        r"RuntimeError.*initialization failed",
        r"Engine core initialization failed",
        r"OutOfMemoryError",
        r"Failed to load model",
        r"Model .* does not exist",
        r"Segmentation fault",
        r"core dumped",
        r"NCCL error",
        r"AssertionError",
        r"torch\.cuda\.OutOfMemoryError",
        r"Cannot allocate memory",
        r"GPU memory allocation failed",
        r"Failed to initialize",
        r"Error loading model weights",
        r"Failed core proc",
        r"MaxRetryError.*huggingface\.co",
        r"Failed to resolve.*huggingface\.co",
        r"NameResolutionError.*huggingface\.co",
        r"Name or service not known.*huggingface\.co",
        r"Invalid repository ID or local directory",
    ]

    def __init__(self):
        self.api_url = os.environ.get("CASOLA_API_URL")
        self.api_token = os.environ.get("CASOLA_API_TOKEN")

        container_id = os.environ.get("CONTAINER_ID", str(os.getpid()))
        self.instance_id = os.environ.get("CASOLA_INSTANCE_ID", f"vast-{container_id}")
        self.queue_id = os.environ.get("CASOLA_QUEUE_ID", "vllm-queue")
        self.worker_id = self.instance_id

        self.heartbeat_interval = int(os.environ.get("CASOLA_HEARTBEAT_INTERVAL", "60"))
        self.max_jobs = int(os.environ.get("CASOLA_MAX_JOBS", "1"))
        self.lease_seconds = int(os.environ.get("CASOLA_LEASE_SECONDS", "30"))
        self.shutdown_grace_period = float(os.environ.get("CASOLA_SHUTDOWN_GRACE_PERIOD", "5.0"))

        # Read config_id from environment (set by scheduler)
        self.config_id = os.environ.get("CASOLA_CONFIG_ID", "")
        if not self.config_id:
            raise ValueError("CASOLA_CONFIG_ID environment variable is required")

        # Backpressure config
        self.backpressure_check_interval = int(
            os.environ.get("CASOLA_BACKPRESSURE_CHECK_INTERVAL", "5")
        )
        self.backpressure_pause_threshold = int(
            os.environ.get("CASOLA_BACKPRESSURE_PAUSE_THRESHOLD", "0")
        )
        self.backpressure_resume_threshold = int(
            os.environ.get("CASOLA_BACKPRESSURE_RESUME_THRESHOLD", "0")
        )

        # Queue mode: always WebSocket
        self.queue_mode = "ws"

        vllm_host = os.environ.get("VLLM_HOST", "127.0.0.1")
        vllm_port = os.environ.get("VLLM_PORT", "8000")

        # Store root URL for health checks and base URL for API calls
        self.vllm_root_url = f"http://{vllm_host}:{vllm_port}"
        self.vllm_base_url = f"{self.vllm_root_url}/v1"
        self.vllm_startup_timeout = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "600"))

        self.running = True
        self.last_heartbeat = 0
        self.current_job_id = None
        self.job_lock = threading.Lock()

        self.transport: Optional[QueueTransport] = None

        self.vllm_client = None
        self.vllm_process = None
        self.vllm_log_monitor_thread = None
        self.vllm_served_model_name = None
        self.error_detected = False
        self.error_message = None
        self.error_is_fatal = False

        # Compile regex patterns for efficiency
        self.compiled_fatal_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.FATAL_ERROR_PATTERNS
        ]
        self.compiled_retryable_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.RETRYABLE_ERROR_PATTERNS
        ]

        # Log shipping
        self.log_buffer: List[Dict[str, Any]] = []
        self.log_lock = threading.Lock()
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
        """Log a message from the worker or subprocess (vLLM/ComfyUI)."""
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

            # Take up to 1000 logs (prevent unbounded memory growth)
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
            # Re-add logs to buffer front (will be retried on next flush)
            with self.log_lock:
                self.log_buffer = batch + self.log_buffer

                # Prevent unbounded growth - drop oldest if buffer too large
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

    def _check_log_for_errors(self, line: str) -> Optional[tuple[str, bool]]:
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

    def _monitor_vllm_logs(self, process: subprocess.Popen):
        """
        Monitor vLLM process logs in real-time and detect fatal errors.
        Runs in a separate thread.
        """
        self.log("Starting vLLM log monitor thread")

        try:
            while self.running and process.poll() is None:
                # Read from stdout (stderr is redirected to stdout)
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        line_str = line.strip()
                        if line_str:
                            # Determine log level from vLLM output patterns
                            level = "info"
                            if "ERROR" in line_str or "Error" in line_str:
                                level = "error"
                            elif "WARNING" in line_str or "Warning" in line_str:
                                level = "warning"

                            # Ship to platform (via log() method which buffers and prints)
                            self.log(line_str, level=level, source="vllm")

                            # Check for error patterns
                            error_match = self._check_log_for_errors(line_str)
                            if error_match and not self.error_detected:
                                matched_pattern, is_fatal = error_match
                                self.error_detected = True
                                self.error_is_fatal = is_fatal
                                error_type = "FATAL" if is_fatal else "RETRYABLE"
                                self.error_message = f"{error_type} vLLM error detected (pattern: {matched_pattern}): {line_str}"
                                self.log(f"ERROR DETECTED: {self.error_message}", level="error")

                                # Send appropriate error heartbeat
                                status = "fatal" if is_fatal else "error"
                                self.send_heartbeat(status)

                                # Trigger shutdown
                                self.log(
                                    f"Initiating worker shutdown due to {error_type} vLLM error",
                                    level="error",
                                )
                                self.running = False

                                # Terminate vLLM process
                                try:
                                    process.terminate()
                                except Exception:
                                    pass

                                break
        except Exception as e:
            self.log(f"Error in vLLM log monitor: {e}", level="error")

        self.log("vLLM log monitor thread exiting")

    def validate_config(self):
        ws_url = os.environ.get("CASOLA_WS_URL")
        if not ws_url:
            self.log("ERROR: CASOLA_WS_URL environment variable is required")
            return False
        return True

    def _start_vllm_process(self) -> bool:
        """
        Start vLLM server as a subprocess with log monitoring.
        Returns True if process started successfully, False otherwise.
        """
        try:
            vllm_model = os.environ.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf")
            vllm_host = os.environ.get("VLLM_HOST", "127.0.0.1")
            vllm_port = os.environ.get("VLLM_PORT", "8000")
            vllm_extra_args = os.environ.get("VLLM_EXTRA_ARGS", "")
            vllm_served_model_name = os.environ.get("VLLM_SERVED_MODEL_NAME", "default")

            self.vllm_served_model_name = vllm_served_model_name

            vllm_serve_command = os.environ.get("VLLM_SERVE_COMMAND", "vllm")

            cmd = [
                vllm_serve_command,
                "serve",
                vllm_model,
                "--host",
                vllm_host,
                "--port",
                vllm_port,
                "--served-model-name",
                vllm_served_model_name,
            ]

            # Add extra args if provided
            if vllm_extra_args:
                cmd.extend(vllm_extra_args.split())

            self.log(f"Starting vLLM process: {' '.join(cmd)}")

            # Start vLLM with stderr redirected to stdout for unified log monitoring
            # This ensures we capture all output in order
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )

            self.log(f"vLLM process started (PID: {self.vllm_process.pid})")

            # Start log monitoring thread immediately
            self.vllm_log_monitor_thread = threading.Thread(
                target=self._monitor_vllm_logs, args=(self.vllm_process,), daemon=True
            )
            self.vllm_log_monitor_thread.start()

            return True

        except Exception as e:
            self.log(f"Failed to start vLLM process: {e}")
            return False

    def check_vllm_health(self) -> bool:
        """
        Checks if vLLM /health endpoint returns 200.
        """
        try:
            response = requests.get(f"{self.vllm_root_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def wait_for_vllm_readiness(self):
        """
        Waits specifically for /health to return 200, then initializes client.
        """
        self.log(f"Waiting for vLLM health check at {self.vllm_root_url}/health...")
        start_time = time.time()

        # 1. Wait for /health 200
        while time.time() - start_time < self.vllm_startup_timeout:
            if self.check_vllm_health():
                self.log("vLLM health check passed (200 OK).")
                break
            time.sleep(2)
        else:
            self.log(f"ERROR: vLLM health check failed within {self.vllm_startup_timeout}s")
            return False

        # 2. Wait for /models (API readiness) and init client
        self.log(f"Initializing vLLM client at {self.vllm_base_url}...")
        try:
            # We verify /models just to ensure the API side is fully responsive
            response = requests.get(f"{self.vllm_base_url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                self.log(f"vLLM API ready. Available models: {models}")
                self.vllm_client = OpenAI(base_url=self.vllm_base_url, api_key="EMPTY")
                return True
        except Exception as e:
            self.log(f"ERROR: Failed to initialize vLLM client: {e}")

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

    def execute_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        if not self.vllm_client:
            raise Exception("vLLM client not initialized")

        self.log(f"Executing chat completion with {len(messages)} messages")

        completion_params = {
            "model": self.vllm_served_model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": False,
        }

        if "max_tokens" in kwargs:
            completion_params["max_tokens"] = kwargs["max_tokens"]

        if "frequency_penalty" in kwargs:
            completion_params["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            completion_params["presence_penalty"] = kwargs["presence_penalty"]
        if "stop" in kwargs:
            completion_params["stop"] = kwargs["stop"]

        response = self.vllm_client.chat.completions.create(**completion_params)

        result = {
            "id": response.id,
            "model": self.vllm_served_model_name,
            "choices": [
                {
                    "index": choice.index,
                    "message": {"role": choice.message.role, "content": choice.message.content},
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

        return result

    def execute_embeddings(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        if not self.vllm_client:
            raise Exception("vLLM client not initialized")

        # Handle both single string and array of strings
        inputs = input_data if isinstance(input_data, list) else [input_data]

        self.log(f"Executing embeddings for {len(inputs)} input(s)")

        try:
            embedding_params = {
                "model": self.vllm_served_model_name,
                "input": inputs,
            }

            # Add optional parameters if provided
            if "encoding_format" in kwargs:
                embedding_params["encoding_format"] = kwargs["encoding_format"]
            if "dimensions" in kwargs:
                embedding_params["dimensions"] = kwargs["dimensions"]

            response = self.vllm_client.embeddings.create(**embedding_params)

            result = {
                "embeddings": [
                    {"index": emb.index, "embedding": emb.embedding} for emb in response.data
                ],
                "prompt_tokens": response.usage.prompt_tokens if hasattr(response, "usage") else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response, "usage") else 0,
            }

            self.log(f"Embeddings completed, {len(result['embeddings'])} embedding(s) generated")
            return result

        except Exception as e:
            self.log(f"Embeddings execution failed: {e}")
            raise

    def execute_tts(self, input_text: str, voice: str, **kwargs) -> Dict[str, Any]:
        if not self.vllm_client:
            raise Exception("vLLM client not initialized")

        self.log(f"Executing TTS for text: {input_text[:50]}...")

        try:
            response_format = kwargs.get("response_format", "mp3")
            speed = kwargs.get("speed", 1.0)

            tts_params = {
                "model": self.vllm_served_model_name,
                "input": input_text,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
            }

            response = self.vllm_client.audio.speech.create(**tts_params)

            audio_bytes = response.content if hasattr(response, "content") else response.read()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            result = {
                "audio_base64": audio_base64,
                "format": response_format,
                "voice": voice,
                "input_length": len(input_text),
            }

            self.log(f"TTS completed, audio size: {len(audio_bytes)} bytes")
            return result

        except Exception as e:
            self.log(f"TTS execution failed: {e}")
            raise

    def execute_transcription(self, audio_base64: str, **kwargs) -> Dict[str, Any]:
        if not self.vllm_client:
            raise Exception("vLLM client not initialized")

        self.log("Executing audio transcription...")

        try:
            import tempfile

            # Decode base64 to bytes
            audio_bytes = base64.b64decode(audio_base64)

            # Write to temp file (OpenAI client requires file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                transcription_params = {
                    "model": self.vllm_served_model_name,
                    "file": open(tmp_path, "rb"),
                    "response_format": kwargs.get("response_format", "json"),
                }

                if "language" in kwargs:
                    transcription_params["language"] = kwargs["language"]
                if "prompt" in kwargs:
                    transcription_params["prompt"] = kwargs["prompt"]
                if "temperature" in kwargs:
                    transcription_params["temperature"] = kwargs["temperature"]

                response = self.vllm_client.audio.transcriptions.create(**transcription_params)

                # Parse response based on format
                if isinstance(response, dict):
                    result = response
                else:
                    # For text format, wrap in expected structure
                    result = {"text": str(response)}

                self.log(f"Transcription completed: {result.get('text', '')[:100]}...")
                return result

            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            self.log(f"Transcription execution failed: {e}")
            raise

    def execute_text_to_video(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a video from a text prompt via vLLM-Omni /v1/videos/generations."""
        num_frames = kwargs.get("num_frames", 81)
        width = kwargs.get("width", 1280)
        height = kwargs.get("height", 720)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 7.0)
        fps = kwargs.get("fps", 16)
        seed = kwargs.get("seed")
        negative_prompt = kwargs.get("negative_prompt")

        self.log(f"Executing T2V: prompt={prompt[:80]}..., {width}x{height}, {num_frames} frames")

        request_body: Dict[str, Any] = {
            "model": self.vllm_served_model_name,
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "fps": fps,
            "response_format": "b64_json",
        }
        if negative_prompt:
            request_body["negative_prompt"] = negative_prompt
        if seed is not None:
            request_body["seed"] = seed

        start_time = time.time()
        response = requests.post(
            f"{self.vllm_base_url}/videos/generations",
            json=request_body,
            timeout=900,
        )
        response.raise_for_status()

        inference_time = time.time() - start_time
        data = response.json()
        video_b64 = data["data"][0]["b64_json"]
        seed_used = data["data"][0].get("seed")

        video_bytes = base64.b64decode(video_b64)
        self.log(f"T2V completed: {len(video_bytes)} bytes in {inference_time:.1f}s")

        return {
            "video_b64": video_b64,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "inference_time_seconds": inference_time,
            "seed_used": seed_used,
        }

    def execute_image_to_video(self, image_url: str, prompt: str = "", **kwargs) -> Dict[str, Any]:
        """Generate a video from a source image via vLLM-Omni I2V pipeline."""
        num_frames = kwargs.get("num_frames", 81)
        width = kwargs.get("width", 1280)
        height = kwargs.get("height", 720)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 7.0)
        fps = kwargs.get("fps", 16)
        seed = kwargs.get("seed")
        negative_prompt = kwargs.get("negative_prompt")

        self.log(f"Executing I2V: image_url={image_url}, {width}x{height}, {num_frames} frames")

        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        img_b64 = base64.b64encode(img_response.content).decode("utf-8")

        content_type = img_response.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if content_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
            content_type = "image/jpeg"

        request_body: Dict[str, Any] = {
            "model": self.vllm_served_model_name,
            "prompt": prompt or "Animate this image naturally",
            "n": 1,
            "size": f"{width}x{height}",
            "image": f"data:{content_type};base64,{img_b64}",
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "fps": fps,
            "response_format": "b64_json",
        }
        if negative_prompt:
            request_body["negative_prompt"] = negative_prompt
        if seed is not None:
            request_body["seed"] = seed

        start_time = time.time()
        response = requests.post(
            f"{self.vllm_base_url}/videos/generations",
            json=request_body,
            timeout=900,
        )
        response.raise_for_status()

        inference_time = time.time() - start_time
        data = response.json()
        video_b64 = data["data"][0]["b64_json"]
        seed_used = data["data"][0].get("seed")

        video_bytes = base64.b64decode(video_b64)
        self.log(f"I2V completed: {len(video_bytes)} bytes in {inference_time:.1f}s")

        return {
            "video_b64": video_b64,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": num_frames / fps,
            "inference_time_seconds": inference_time,
            "seed_used": seed_used,
        }

    def process_job(self, job: Dict[str, Any]) -> bool:
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
            # Route by task field from the payload (not config_id pattern)
            if task == "openai/audio-speech":
                input_text = payload.get("input", "")
                voice = payload.get("voice", "default")
                if not input_text:
                    raise ValueError("No input text provided in payload")

                tts_kwargs = {
                    "model": self.vllm_served_model_name,
                    "response_format": payload.get("response_format", "mp3"),
                    "speed": payload.get("speed", 1.0),
                }

                tts_result = self.execute_tts(input_text, voice, **tts_kwargs)

                result_data = {
                    "status": "success",
                    "config_id": config_id,
                    "task": task,
                    "audio_base64": tts_result["audio_base64"],
                    "format": tts_result["format"],
                    "voice": tts_result["voice"],
                    "input_length": tts_result["input_length"],
                    "processed_at": datetime.now().isoformat(),
                }

                result = self.transport.complete_job(job_id, result_data)
            elif task == "openai/audio-transcription":
                audio_base64 = payload.get("audio_base64")
                audio_url = payload.get("audio_url")

                if not audio_base64 and not audio_url:
                    raise ValueError("No audio_base64 or audio_url provided in payload")

                # Download from URL if provided
                if audio_url:
                    resp = requests.get(audio_url, timeout=30)
                    resp.raise_for_status()
                    audio_base64 = base64.b64encode(resp.content).decode("utf-8")

                stt_kwargs = {
                    "response_format": payload.get("response_format", "json"),
                }
                if "language" in payload:
                    stt_kwargs["language"] = payload["language"]
                if "prompt" in payload:
                    stt_kwargs["prompt"] = payload["prompt"]
                if "temperature" in payload:
                    stt_kwargs["temperature"] = payload["temperature"]

                stt_result = self.execute_transcription(audio_base64, **stt_kwargs)

                result_data = {
                    "status": "success",
                    "config_id": config_id,
                    "task": task,
                    "text": stt_result.get("text", ""),
                    "processed_at": datetime.now().isoformat(),
                }

                # Add optional fields if present
                if "language" in stt_result:
                    result_data["language"] = stt_result["language"]
                if "duration" in stt_result:
                    result_data["duration"] = stt_result["duration"]
                if "segments" in stt_result:
                    result_data["segments"] = stt_result["segments"]

                result = self.transport.complete_job(job_id, result_data)
            elif task == "openai/embeddings":
                input_data = payload.get("input")
                if not input_data:
                    raise ValueError("No input provided in payload")

                embeddings_kwargs = {}
                if "encoding_format" in payload:
                    embeddings_kwargs["encoding_format"] = payload["encoding_format"]
                if "dimensions" in payload:
                    embeddings_kwargs["dimensions"] = payload["dimensions"]

                embeddings_result = self.execute_embeddings(input_data, **embeddings_kwargs)

                result_data = {
                    "status": "success",
                    "config_id": config_id,
                    "task": task,
                    "embeddings": embeddings_result["embeddings"],
                    "prompt_tokens": embeddings_result["prompt_tokens"],
                    "total_tokens": embeddings_result["total_tokens"],
                    "processed_at": datetime.now().isoformat(),
                }

                result = self.transport.complete_job(job_id, result_data)
            elif task in ("openai/chat-completion", ""):
                # Chat completion: explicit task or empty (legacy default)
                messages = payload.get("messages", [])
                if not messages:
                    raise ValueError("No messages provided in payload")

                completion_kwargs = {
                    "model": self.vllm_served_model_name,
                    "temperature": payload.get("temperature", 0.7),
                    "top_p": payload.get("top_p", 1.0),
                }

                if "max_tokens" in payload:
                    completion_kwargs["max_tokens"] = payload["max_tokens"]

                if "frequency_penalty" in payload:
                    completion_kwargs["frequency_penalty"] = payload["frequency_penalty"]
                if "presence_penalty" in payload:
                    completion_kwargs["presence_penalty"] = payload["presence_penalty"]
                if "stop" in payload:
                    completion_kwargs["stop"] = payload["stop"]

                completion_result = self.execute_chat_completion(messages, **completion_kwargs)

                result_data = {
                    "status": "success",
                    "config_id": config_id,
                    "task": task or "openai/chat-completion",
                    "completion": completion_result,
                    "processed_at": datetime.now().isoformat(),
                }

                result = self.transport.complete_job(job_id, result_data)
            elif task == "fal/text-to-video":
                prompt = payload.get("prompt", "")
                if not prompt:
                    raise ValueError("No prompt provided in payload")

                video_result = self.execute_text_to_video(
                    prompt,
                    negative_prompt=payload.get("negative_prompt"),
                    num_frames=payload.get("num_frames", 81),
                    width=payload.get("width", 1280),
                    height=payload.get("height", 720),
                    fps=payload.get("fps", 16),
                    num_inference_steps=payload.get("num_inference_steps", 50),
                    guidance_scale=payload.get("guidance_scale", 7.0),
                    seed=payload.get("seed"),
                )

                result_data = {
                    "status": "success",
                    "config_id": config_id,
                    "task": task,
                    "video_b64": video_result["video_b64"],
                    "width": video_result["width"],
                    "height": video_result["height"],
                    "num_frames": video_result["num_frames"],
                    "fps": video_result["fps"],
                    "duration_seconds": video_result["duration_seconds"],
                    "inference_time_seconds": video_result["inference_time_seconds"],
                    "seed_used": video_result.get("seed_used"),
                    "processed_at": datetime.now().isoformat(),
                }

                result = self.transport.complete_job(job_id, result_data)

            elif task == "fal/image-to-video":
                image_url = payload.get("image_url", "")
                if not image_url:
                    raise ValueError("No image_url provided in payload")

                video_result = self.execute_image_to_video(
                    image_url,
                    prompt=payload.get("prompt", ""),
                    negative_prompt=payload.get("negative_prompt"),
                    num_frames=payload.get("num_frames", 81),
                    width=payload.get("width", 1280),
                    height=payload.get("height", 720),
                    fps=payload.get("fps", 16),
                    num_inference_steps=payload.get("num_inference_steps", 50),
                    guidance_scale=payload.get("guidance_scale", 7.0),
                    seed=payload.get("seed"),
                )

                result_data = {
                    "status": "success",
                    "config_id": config_id,
                    "task": task,
                    "video_b64": video_result["video_b64"],
                    "width": video_result["width"],
                    "height": video_result["height"],
                    "num_frames": video_result["num_frames"],
                    "fps": video_result["fps"],
                    "duration_seconds": video_result["duration_seconds"],
                    "inference_time_seconds": video_result["inference_time_seconds"],
                    "seed_used": video_result.get("seed_used"),
                    "source_image_url": image_url,
                    "processed_at": datetime.now().isoformat(),
                }

                result = self.transport.complete_job(job_id, result_data)

            elif task == "fal/speech-to-video":
                audio_url = payload.get("audio_url", "")
                image_url = payload.get("image_url", "")
                if not audio_url:
                    raise ValueError("No audio_url provided in payload")
                if not image_url:
                    raise ValueError("No image_url provided in payload")

                # Download audio from URL and encode as base64 data URI
                audio_resp = requests.get(audio_url, timeout=60)
                audio_resp.raise_for_status()
                audio_b64 = base64.b64encode(audio_resp.content).decode("utf-8")
                audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

                # Download image from URL and encode as base64 data URI
                image_resp = requests.get(image_url, timeout=60)
                image_resp.raise_for_status()
                image_b64 = base64.b64encode(image_resp.content).decode("utf-8")
                image_data_uri = f"data:image/jpeg;base64,{image_b64}"

                video_payload = {
                    "model": self.vllm_served_model_name,
                    "audio": audio_data_uri,
                    "image": image_data_uri,
                }
                if payload.get("prompt"):
                    video_payload["prompt"] = payload["prompt"]
                if payload.get("num_inference_steps"):
                    video_payload["num_inference_steps"] = payload["num_inference_steps"]
                if payload.get("guidance_scale"):
                    video_payload["guidance_scale"] = payload["guidance_scale"]
                if payload.get("seed") is not None:
                    video_payload["seed"] = payload["seed"]
                if payload.get("num_frames"):
                    video_payload["num_frames"] = payload["num_frames"]
                if payload.get("fps"):
                    video_payload["fps"] = payload["fps"]

                s2v_resp = requests.post(
                    f"{self.vllm_root_url}/videos/generations",
                    json=video_payload,
                    timeout=600,
                )
                s2v_resp.raise_for_status()
                video_result = s2v_resp.json()

                result_data = {
                    "status": "success",
                    "config_id": config_id,
                    "task": task,
                    "video_b64": video_result["video_b64"],
                    "width": video_result["width"],
                    "height": video_result["height"],
                    "num_frames": video_result["num_frames"],
                    "fps": video_result["fps"],
                    "duration_seconds": video_result["duration_seconds"],
                    "inference_time_seconds": video_result.get("inference_time_seconds", 0),
                    "seed_used": video_result.get("seed_used"),
                    "source_audio_url": audio_url,
                    "source_image_url": image_url,
                    "processed_at": datetime.now().isoformat(),
                }

                result = self.transport.complete_job(job_id, result_data)

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

    def _heartbeat_loop(self):
        # We start this thread AFTER the initial checks in run(), so we just loop.
        while self.running:
            try:
                # Sleep first, as the initial "running" heartbeat was sent in run()
                time.sleep(self.heartbeat_interval)

                if self.running:
                    # Enforce health check before sending heartbeat
                    if self.check_vllm_health():
                        self.send_heartbeat("active")
                    else:
                        self.log("Skipping heartbeat: vLLM health check failed")
            except Exception as e:
                self.log(f"Error in heartbeat loop: {e}")

    def _get_vllm_waiting_requests(self) -> Optional[int]:
        """Fetch vllm:num_requests_waiting from the vLLM Prometheus /metrics endpoint."""
        try:
            response = requests.get(f"{self.vllm_root_url}/metrics", timeout=2)
            if response.status_code != 200:
                return None
            for line in response.text.splitlines():
                if line.startswith("vllm:num_requests_waiting"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(float(parts[-1]))
            return None
        except Exception:
            return None

    def _collect_vllm_metrics(self) -> Dict[str, Any]:
        """Collect vLLM engine metrics for the system metrics reporter."""
        metrics: Dict[str, Any] = {}
        try:
            response = requests.get(f"{self.vllm_root_url}/metrics", timeout=2)
            if response.status_code != 200:
                return metrics
            for line in response.text.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics["vllm_running_requests"] = int(float(parts[-1]))
                elif line.startswith("vllm:num_requests_waiting"):
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics["vllm_waiting_requests"] = int(float(parts[-1]))
        except Exception:
            pass
        return metrics

    def _backpressure_monitor_loop(self):
        """Poll vLLM metrics and pause/resume the transport based on queue depth."""
        while self.running:
            time.sleep(self.backpressure_check_interval)
            if not self.running or not self.transport:
                break
            waiting = self._get_vllm_waiting_requests()
            if waiting is None:
                continue
            if not self.transport.paused and waiting > self.backpressure_pause_threshold:
                self.log(
                    f"Backpressure: pausing (waiting={waiting}, threshold={self.backpressure_pause_threshold})"
                )
                self.transport.pause()
            elif self.transport.paused and waiting <= self.backpressure_resume_threshold:
                self.log(
                    f"Backpressure: resuming (waiting={waiting}, threshold={self.backpressure_resume_threshold})"
                )
                self.transport.resume()

    def _transport_loop(self):
        """Run the transport in a thread (blocking call)."""
        try:
            self.transport.start()
        except Exception:
            self.log("Transport loop exited with error")

    def run(self):
        if not self.validate_config():
            sys.exit(1)

        self.log("Starting vLLM worker...")
        self.log(f"Queue mode: {self.queue_mode}")
        self.log(f"Instance ID: {self.instance_id}")
        self.log(f"Worker ID: {self.worker_id}")
        if self.api_url:
            self.log(f"API URL: {self.api_url}")

        # Print environment variables for debugging
        self.print_environment_variables()

        # 1. Send "Starting" heartbeat immediately
        self.send_heartbeat("starting")

        # 2. Run pre-startup GPU health checks
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

        # 3. Start vLLM process with log monitoring
        if not self._start_vllm_process():
            self.log("Failed to start vLLM process")
            self.send_heartbeat("error")
            sys.exit(1)

        # 4. Wait for vLLM to be healthy (checks /health 200)
        #    and initialize client
        if not self.wait_for_vllm_readiness():
            self.log("Failed to start: vLLM did not become healthy.")
            # If an error was detected during startup, use appropriate status
            if self.error_detected:
                status = "fatal" if self.error_is_fatal else "error"
                self.send_heartbeat(status)
            else:
                self.send_heartbeat("failed")
            # Clean up vLLM process
            if self.vllm_process:
                self.vllm_process.terminate()
                self.vllm_process.wait(timeout=5)
            sys.exit(1)

        # 5. Build transport config from env vars
        config = {
            "worker_id": self.worker_id,
            "max_jobs": self.max_jobs,
            "lease_seconds": self.lease_seconds,
            "config_id": self.config_id,
            "ws_url": os.environ.get("CASOLA_WS_URL", ""),
            "capacity": int(os.environ.get("CASOLA_CAPACITY", "1")),
            "api_token": self.api_token,
        }

        self.transport = create_transport(config)
        self.transport.on_job = self.process_job
        self.transport.on_connected = lambda: self.send_heartbeat("active")

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
            engine_metrics_fn=self._collect_vllm_metrics,
            log_fn=self.log,
        )
        self.metrics_reporter.start()

        # 7. Start transport in a thread so main loop can monitor vLLM health
        transport_thread = threading.Thread(target=self._transport_loop, daemon=True)
        transport_thread.start()

        # 8. Start backpressure monitor (if enabled)
        backpressure_thread = None
        if self.backpressure_check_interval > 0:
            backpressure_thread = threading.Thread(
                target=self._backpressure_monitor_loop, daemon=True
            )
            backpressure_thread.start()

        try:
            while self.running:
                # Check if vLLM process died unexpectedly
                if self.vllm_process and self.vllm_process.poll() is not None:
                    if not self.error_detected:
                        self.log("vLLM process died unexpectedly")
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
        if backpressure_thread:
            backpressure_thread.join(timeout=5)

        # Clean up vLLM process
        if self.vllm_process:
            self.log("Terminating vLLM process")
            try:
                self.vllm_process.terminate()
                self.vllm_process.wait(timeout=5)
            except Exception:
                self.log("Force killing vLLM process")
                self.vllm_process.kill()

        # Exit with error code if error was detected
        if self.error_detected:
            sys.exit(1)


if __name__ == "__main__":
    worker = VLLMWorker()
    worker.run()
