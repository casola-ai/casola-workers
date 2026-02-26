"""End-to-end tests for the Casola test server."""

from __future__ import annotations

import time

import httpx

from .conftest import MockWorker


class TestChatCompletion:
    def test_chat_completion_roundtrip(
        self, server: str, mock_worker: MockWorker, client: httpx.Client
    ):
        """Worker registers, client sends chat completion, gets OpenAI response."""

        def handle_job(msg):
            payload = msg.get("payload", {})
            messages = payload.get("messages", [])
            last = messages[-1]["content"] if messages else ""
            return {
                "status": "success",
                "task": "openai/chat-completion",
                "completion": {
                    "id": f"chatcmpl-{msg['job_id']}",
                    "model": payload.get("model", "test"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"Echo: {last}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                },
            }

        mock_worker.on_job = handle_job
        mock_worker.start()
        time.sleep(0.3)  # Let registration complete

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] == "Echo: hello"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_no_worker_returns_503(self, server: str, client: httpx.Client):
        """When no worker is connected, return 503."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 503

    def test_worker_error_returns_502(
        self, server: str, mock_worker: MockWorker, client: httpx.Client
    ):
        """When the worker returns an error, relay it as 502."""
        mock_worker.on_job = lambda msg: "Simulated GPU OOM"
        mock_worker.start()
        time.sleep(0.3)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 502
        assert "Simulated GPU OOM" in resp.json()["detail"]


class TestEmbeddings:
    def test_embeddings_roundtrip(self, server: str, mock_worker: MockWorker, client: httpx.Client):
        def handle_job(msg):
            return {
                "status": "success",
                "task": "openai/embeddings",
                "embeddings": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "prompt_tokens": 3,
                "total_tokens": 3,
            }

        mock_worker.on_job = handle_job
        mock_worker.start()
        time.sleep(0.3)

        resp = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "hello world",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]


class TestPersistedJobs:
    def test_persisted_job_lifecycle(
        self, server: str, mock_worker: MockWorker, client: httpx.Client
    ):
        """Create job, poll pending, worker completes, poll completed."""
        # Create a job before the worker connects
        resp = client.post(
            "/jobs",
            json={
                "payload": {
                    "task": "openai/chat-completion",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            },
        )
        assert resp.status_code == 200
        job = resp.json()
        job_id = job["id"]
        assert job["status"] == "pending"

        # Poll — still pending (no worker yet)
        resp = client.get(f"/jobs/{job_id}")
        assert resp.json()["status"] == "pending"

        # Now connect a worker that auto-handles the backlog
        mock_worker.on_job = lambda msg: {"status": "success", "answer": 42}
        mock_worker.start()
        time.sleep(0.5)  # Wait for backlog drain and processing

        # Poll — should be completed
        resp = client.get(f"/jobs/{job_id}")
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"]["answer"] == 42

    def test_job_not_found(self, server: str, client: httpx.Client):
        resp = client.get("/jobs/nonexistent-id")
        assert resp.status_code == 404


class TestHealth:
    def test_health_endpoint(self, server: str, client: httpx.Client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "worker_connected" in data
        assert "pending_jobs" in data


class TestModels:
    def test_models_empty(self, server: str, client: httpx.Client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"

    def test_models_with_worker(self, server: str, mock_worker: MockWorker, client: httpx.Client):
        mock_worker.on_job = lambda msg: {}
        mock_worker.start()
        time.sleep(0.3)

        resp = client.get("/v1/models")
        data = resp.json()
        config_ids = [m["id"] for m in data["data"]]
        assert "default" in config_ids
