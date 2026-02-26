"""Pytest fixtures: server lifecycle, worker connection helper, HTTP client."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Generator

import httpx
import pytest
import uvicorn
import websockets.sync.client
from casola_test_server.app import create_app

# Allow importing the casola_worker SDK from the workers/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# =============================================================================
# Server fixture — runs FastAPI in a background thread
# =============================================================================

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18788  # Use a non-standard port to avoid conflicts


class _ServerThread(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


@pytest.fixture(scope="session")
def server() -> Generator[str, None, None]:
    """Start the test server in a background thread, yield the base URL."""
    app = create_app()
    thread = _ServerThread(app)
    thread.start()

    base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"

    # Wait for server to be ready
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.05)
    else:
        raise RuntimeError("Test server did not start in time")

    yield base_url

    thread.stop()
    thread.join(timeout=5)


# =============================================================================
# Worker connection helper — connects a mock worker via WebSocket
# =============================================================================


class MockWorker:
    """A lightweight mock worker that connects via WebSocket and processes jobs."""

    def __init__(
        self,
        ws_url: str,
        worker_id: str = "test-worker",
        config_id: str = "default",
        capacity: int = 1,
    ):
        self.ws_url = ws_url
        self.worker_id = worker_id
        self.config_id = config_id
        self.capacity = capacity
        self._ws = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._jobs_received: list[dict] = []
        self._lock = threading.Lock()
        self.on_job = None  # Callback: (job_dict) -> dict | str (result or error string)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def jobs_received(self) -> list[dict]:
        with self._lock:
            return list(self._jobs_received)

    def _run(self):
        ws_endpoint = self.ws_url.rstrip("/") + "/ws"
        self._ws = websockets.sync.client.connect(ws_endpoint, open_timeout=5)

        # Register
        self._ws.send(
            json.dumps(
                {
                    "type": "register",
                    "worker_id": self.worker_id,
                    "config_id": self.config_id,
                    "capacity": self.capacity,
                }
            )
        )

        while self._running:
            try:
                raw = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            except Exception:
                break

            msg = json.loads(raw)
            if msg.get("type") == "job":
                with self._lock:
                    self._jobs_received.append(msg)

                job_id = msg["job_id"]
                fence_token = msg.get("fence_token", 0)

                if self.on_job:
                    result = self.on_job(msg)
                    if isinstance(result, str):
                        # String = error
                        self._ws.send(
                            json.dumps(
                                {
                                    "type": "job_error",
                                    "job_id": job_id,
                                    "error": result,
                                    "fence_token": fence_token,
                                }
                            )
                        )
                    else:
                        self._ws.send(
                            json.dumps(
                                {
                                    "type": "job_result",
                                    "job_id": job_id,
                                    "result": result,
                                    "fence_token": fence_token,
                                }
                            )
                        )

    def wait_for_jobs(self, count: int = 1, timeout: float = 5.0) -> list[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.jobs_received) >= count:
                return self.jobs_received[:count]
            time.sleep(0.05)
        raise TimeoutError(f"Expected {count} jobs, got {len(self.jobs_received)}")


@pytest.fixture
def mock_worker(server: str):
    """Create and yield a mock worker connected to the test server. Stops on cleanup."""
    ws_url = server.replace("http://", "ws://")
    worker = MockWorker(ws_url)
    yield worker
    worker.stop()


# =============================================================================
# HTTP client fixture
# =============================================================================


@pytest.fixture
def client(server: str) -> httpx.Client:
    """Synchronous httpx client pointed at the test server."""
    return httpx.Client(base_url=server, timeout=30)
