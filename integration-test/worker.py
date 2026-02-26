#!/usr/bin/env python3

import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

# Allow importing casola_worker from parent directory (local dev)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from casola_worker.transport import QueueTransport, create_transport


class WorkerIntegrationTest:
    def __init__(self):
        self.api_url = os.environ.get("CASOLA_API_URL")
        self.api_token = os.environ.get("CASOLA_API_TOKEN")

        container_id = os.environ.get("CONTAINER_ID", str(os.getpid()))
        # Auto-detect provider from environment:
        # - RunPod sets RUNPOD_POD_ID automatically on all pods
        # - Vast.ai uses CONTAINER_ID or falls back to PID
        # - CASOLA_INSTANCE_ID always takes priority if explicitly set
        runpod_pod_id = os.environ.get("RUNPOD_POD_ID")
        if runpod_pod_id:
            default_id = f"runpod-{runpod_pod_id}"
        else:
            default_id = f"vast-{container_id}"
        self.instance_id = os.environ.get("CASOLA_INSTANCE_ID", default_id)
        self.queue_id = os.environ.get("CASOLA_QUEUE_ID", "test-queue")
        self.worker_id = self.instance_id

        self.heartbeat_interval = int(os.environ.get("CASOLA_HEARTBEAT_INTERVAL", "60"))
        self.max_jobs = int(os.environ.get("CASOLA_MAX_JOBS", "1"))
        self.lease_seconds = int(os.environ.get("CASOLA_LEASE_SECONDS", "30"))
        self.shutdown_grace_period = float(os.environ.get("CASOLA_SHUTDOWN_GRACE_PERIOD", "5.0"))

        # Read config_id from environment (set by scheduler)
        self.config_id = os.environ.get("CASOLA_CONFIG_ID", "")
        if not self.config_id:
            raise ValueError("CASOLA_CONFIG_ID environment variable is required")

        self.simulated_execution_time = float(
            os.environ.get("CASOLA_SIMULATED_EXECUTION_TIME", "2.0")
        )
        self.simulated_error_rate = float(os.environ.get("CASOLA_SIMULATED_ERROR_RATE", "0.0"))

        # Queue mode: always WebSocket
        self.queue_mode = "ws"

        self.running = True
        self.last_heartbeat = 0
        self.current_job_id = None
        self.job_lock = threading.Lock()

        self.transport: Optional[QueueTransport] = None

        # Log shipping
        self.log_buffer: List[Dict[str, Any]] = []
        self.log_lock_logs = threading.Lock()
        self.last_log_flush = time.time()
        self.log_flush_interval = int(os.environ.get("CASOLA_LOG_FLUSH_INTERVAL", "30"))
        self.ws_url = os.environ.get("CASOLA_WS_URL", "")

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
        """Log a message from the worker."""
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

    def _api_call(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an API call and log its duration."""
        start_time = time.time()
        try:
            response = requests.request(method, url, **kwargs)
            duration = time.time() - start_time
            self.log(f"API {method} {url} - {response.status_code} - {duration * 1000:.0f}ms")
            return response
        except Exception as e:
            duration = time.time() - start_time
            self.log(f"API {method} {url} - ERROR: {e} - {duration * 1000:.0f}ms")
            raise

    def validate_config(self):
        ws_url = os.environ.get("CASOLA_WS_URL")
        if not ws_url:
            self.log("ERROR: CASOLA_WS_URL environment variable is required")
            return False
        return True

    def send_heartbeat(self, status: str = "running") -> bool:
        if not self.api_url or not self.api_token:
            return True  # Heartbeats are optional in WS mode
        try:
            payload = {"instance_id": self.instance_id, "queue_id": self.queue_id, "status": status}

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }

            self.log(f"Sending heartbeat: status={status}")
            response = self._api_call(
                "POST", f"{self.api_url}/api/heartbeats", json=payload, headers=headers, timeout=10
            )

            if response.status_code >= 400:
                self.log(f"Heartbeat failed with status {response.status_code}: {response.text}")
                return False

            self.last_heartbeat = time.time()
            return True

        except Exception as e:
            self.log(f"Failed to send heartbeat: {e}")
            return False

    def process_job(self, job: Dict[str, Any]) -> bool:
        job_id = job.get("id")
        config_id = job.get("config_id")
        payload = job.get("payload", {})
        mode = job.get("mode", "persisted")

        # Extract task from payload (full task ID, e.g. 'openai/chat-completion').
        # The API always injects task into the payload before forwarding to the queue.
        task = payload.get("task", "")

        with self.job_lock:
            self.current_job_id = job_id

        self.log(f"Processing job {job_id} (config_id: {config_id}, task: {task}, mode: {mode})")

        try:
            import random

            time.sleep(self.simulated_execution_time)

            if random.random() < self.simulated_error_rate:
                error_msg = f"Simulated error (error rate: {self.simulated_error_rate})"
                self.log(f"Simulating error for job {job_id}: {error_msg}")
                result = self.transport.fail_job(job_id, error_msg)
            else:
                # Handle TTS tasks with mock audio response
                if task == "openai/audio-speech":
                    import base64

                    input_text = payload.get("input", "")
                    voice = payload.get("voice", "default")
                    response_format = payload.get("response_format", "mp3")

                    # Generate a minimal valid WAV file (44-byte header + silence)
                    # so the frontend can render an <audio> element from the blob URL.
                    import struct

                    num_samples = 1000  # ~0.023s of silence at 44100 Hz
                    data_size = num_samples * 2  # 16-bit mono
                    wav_header = struct.pack(
                        "<4sI4s4sIHHIIHH4sI",
                        b"RIFF",
                        36 + data_size,
                        b"WAVE",
                        b"fmt ",
                        16,
                        1,
                        1,
                        44100,
                        88200,
                        2,
                        16,
                        b"data",
                        data_size,
                    )
                    fake_audio = wav_header + b"\x00" * data_size
                    audio_b64 = base64.b64encode(fake_audio).decode("utf-8")

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "audio_base64": audio_b64,
                        "format": response_format,
                        "voice": voice,
                        "input_length": len(input_text),
                        "processed_at": datetime.now().isoformat(),
                    }
                # Handle chat-completion tasks with mock OpenAI response
                elif task == "openai/chat-completion":
                    messages = payload.get("messages", [])
                    last_message = messages[-1].get("content", "") if messages else ""

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "completion": {
                            "id": f"chatcmpl-mock-{job_id}",
                            "model": job.get("model_id", "mock-model"),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": f"Mock response to: {last_message[:500]}...",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(last_message.split()) if last_message else 0,
                                "completion_tokens": 10,
                                "total_tokens": len(last_message.split()) + 10
                                if last_message
                                else 10,
                            },
                        },
                        "processed_at": datetime.now().isoformat(),
                    }
                # Handle audio-transcription tasks with mock STT response
                elif task == "openai/audio-transcription":
                    response_format = payload.get("response_format", "json")
                    language = payload.get("language", "en")

                    # Mock transcription result
                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "text": f"Mock transcription result for job {job_id}. This is a simulated STT output.",
                        "language": language,
                        "duration": 2.5,
                        "segments": [
                            {
                                "start": 0.0,
                                "end": 2.5,
                                "text": f"Mock transcription result for job {job_id}. This is a simulated STT output.",
                            }
                        ],
                        "processed_at": datetime.now().isoformat(),
                    }
                # Handle embeddings tasks with mock response
                elif task == "openai/embeddings":
                    input_data = payload.get("input")
                    if not input_data:
                        input_data = []

                    # Handle both single string and array of strings
                    inputs = input_data if isinstance(input_data, list) else [input_data]

                    # Generate mock embeddings (384-dimensional vectors)
                    dimensions = payload.get("dimensions", 384)
                    embeddings_list = []
                    for idx, text in enumerate(inputs):
                        # Simple deterministic mock embedding based on text length
                        import hashlib

                        hash_obj = hashlib.md5(str(text).encode())
                        hash_int = int(hash_obj.hexdigest(), 16)

                        # Generate pseudo-random but deterministic embedding
                        embedding = [(hash_int + i) % 1000 / 1000.0 for i in range(dimensions)]
                        embeddings_list.append({"index": idx, "embedding": embedding})

                    # Calculate token counts
                    total_text = " ".join(str(t) for t in inputs)
                    prompt_tokens = len(total_text.split())

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "embeddings": embeddings_list,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens,
                        "processed_at": datetime.now().isoformat(),
                    }
                # Handle text-to-video tasks with mock Wan 2.2 response
                elif task == "fal/text-to-video":
                    prompt = payload.get("prompt", "")
                    num_frames = payload.get("num_frames", 81)
                    width = payload.get("width", 1280)
                    height = payload.get("height", 720)
                    fps = payload.get("fps", 16)
                    seed = payload.get("seed", 42)

                    import base64

                    # Small fake MP4 bytes (not a real video, just for data-path testing)
                    fake_video = f"fake-video-t2v-{job_id}".encode("utf-8")
                    video_b64 = base64.b64encode(fake_video).decode("utf-8")

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "video_b64": video_b64,
                        "width": width,
                        "height": height,
                        "num_frames": num_frames,
                        "fps": fps,
                        "duration_seconds": num_frames / fps,
                        "inference_time_seconds": self.simulated_execution_time,
                        "seed_used": seed,
                        "processed_at": datetime.now().isoformat(),
                    }

                # Handle image-to-video tasks with mock Wan 2.2 response
                elif task == "fal/image-to-video":
                    prompt = payload.get("prompt", "")
                    image_url = payload.get("image_url", "")
                    num_frames = payload.get("num_frames", 81)
                    width = payload.get("width", 1280)
                    height = payload.get("height", 720)
                    fps = payload.get("fps", 16)
                    seed = payload.get("seed", 42)

                    import base64

                    fake_video = f"fake-video-i2v-{job_id}".encode("utf-8")
                    video_b64 = base64.b64encode(fake_video).decode("utf-8")

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "video_b64": video_b64,
                        "width": width,
                        "height": height,
                        "num_frames": num_frames,
                        "fps": fps,
                        "duration_seconds": num_frames / fps,
                        "inference_time_seconds": self.simulated_execution_time,
                        "seed_used": seed,
                        "source_image_url": image_url,
                        "processed_at": datetime.now().isoformat(),
                    }

                # Handle speech-to-video tasks with mock Wan 2.2 S2V response
                elif task == "fal/speech-to-video":
                    audio_url = payload.get("audio_url", "")
                    image_url = payload.get("image_url", "")
                    prompt = payload.get("prompt", "")
                    num_frames = payload.get("num_frames", 81)
                    width = payload.get("width", 1024)
                    height = payload.get("height", 704)
                    fps = payload.get("fps", 24)
                    seed = payload.get("seed", 42)

                    import base64

                    fake_video = f"fake-video-s2v-{job_id}".encode("utf-8")
                    video_b64 = base64.b64encode(fake_video).decode("utf-8")

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "video_b64": video_b64,
                        "width": width,
                        "height": height,
                        "num_frames": num_frames,
                        "fps": fps,
                        "duration_seconds": num_frames / fps,
                        "inference_time_seconds": self.simulated_execution_time,
                        "seed_used": seed,
                        "source_audio_url": audio_url,
                        "source_image_url": image_url,
                        "processed_at": datetime.now().isoformat(),
                    }

                # Handle image editing tasks with mock Fal response
                elif task == "fal/image-edit":
                    image_url = payload.get("image_url", "")
                    prompt = payload.get("prompt", "")
                    num_images = payload.get("num_images", 1)
                    seed = payload.get("seed", 42)

                    images = []
                    for i in range(num_images):
                        images.append(
                            {
                                "url": f"https://mock-cdn.example.com/images/edited-{job_id}-img-{i}.png",
                                "width": 1024,
                                "height": 1024,
                                "content_type": "image/png",
                            }
                        )

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "images": images,
                        "timings": {"inference": self.simulated_execution_time},
                        "seed": seed,
                        "source_image_url": image_url,
                        "prompt": prompt,
                        "processed_at": datetime.now().isoformat(),
                    }

                # Handle text-to-image tasks with mock Fal response
                elif task == "fal/text-to-image":
                    prompt = payload.get("prompt", "")
                    num_images = payload.get("num_images", 1)
                    image_size = payload.get("image_size", "square_hd")
                    seed = payload.get("seed", 42)

                    # Generate mock image URLs and metadata
                    images = []
                    for i in range(num_images):
                        images.append(
                            {
                                "url": f"https://mock-cdn.example.com/images/job-{job_id}-img-{i}.png",
                                "width": 1024,
                                "height": 1024,
                                "content_type": "image/png",
                            }
                        )

                    result_data = {
                        "status": "success",
                        "config_id": config_id,
                        "task": task,
                        "images": images,
                        "timings": {"inference": self.simulated_execution_time},
                        "seed": seed,
                        "prompt": prompt,
                        "image_size": image_size,
                        "processed_at": datetime.now().isoformat(),
                    }
                else:
                    # Generic response for other task types
                    result_data = {
                        "status": "success",
                        "processed_at": datetime.now().isoformat(),
                        "config_id": config_id,
                        "task": task,
                        "execution_time": self.simulated_execution_time,
                        "message": "Job processed successfully by integration test worker",
                    }

                # Support result padding for R2 offload testing.
                # When the payload contains result_padding_bytes (int), pad the
                # result so the serialised JSON exceeds the Queue DO's offload
                # threshold (256 KB), forcing result storage into R2.
                result_padding_bytes = payload.get("result_padding_bytes")
                if isinstance(result_padding_bytes, (int, float)) and result_padding_bytes > 0:
                    result_data["_padding"] = "x" * int(result_padding_bytes)

                result = self.transport.complete_job(job_id, result_data)

            with self.job_lock:
                self.current_job_id = None

            return result

        except Exception as e:
            self.log(f"Error processing job {job_id}: {e}")
            self.transport.fail_job(job_id, str(e))

            with self.job_lock:
                self.current_job_id = None

            return False

    def _heartbeat_loop(self):
        while self.running:
            try:
                time.sleep(self.heartbeat_interval)
                if self.running:
                    self.send_heartbeat("active")
            except Exception as e:
                self.log(f"Error in heartbeat loop: {e}")

    def run(self):
        if not self.validate_config():
            sys.exit(1)

        self.log("Starting worker integration test...")
        self.log(f"Queue mode: {self.queue_mode}")
        self.log(f"Instance ID: {self.instance_id}")
        self.log(f"Worker ID: {self.worker_id}")

        # Print environment variables for debugging
        self.print_environment_variables()

        if self.api_url:
            self.log(f"API URL: {self.api_url}")
        self.log(f"Queue ID: {self.queue_id}")
        self.log(f"Heartbeat interval: {self.heartbeat_interval}s")
        self.log(f"Simulated execution time: {self.simulated_execution_time}s")
        self.log(f"Simulated error rate: {self.simulated_error_rate}")
        self.log(f"Shutdown grace period: {self.shutdown_grace_period}s")

        # Build transport config from env vars
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

        # Start heartbeat thread (optional — needs API URL/token)
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        if self.api_url and self.api_token:
            heartbeat_thread.start()

        # Start log flush thread
        log_flush_thread = threading.Thread(target=self._log_flush_loop, daemon=True)
        log_flush_thread.start()

        try:
            # transport.start() is blocking
            self.transport.start()
        except KeyboardInterrupt:
            self.log("Received keyboard interrupt")
            self.running = False

        if self.api_url and self.api_token:
            heartbeat_thread.join(timeout=5)


if __name__ == "__main__":
    worker = WorkerIntegrationTest()
    worker.run()
