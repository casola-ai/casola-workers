"""Core dispatch engine: single worker slot, transient + persisted dispatch, result relay."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from starlette.websockets import WebSocket

from .models import Job, JobAssignment, JobStatus

logger = logging.getLogger(__name__)

try:
    import uuid_utils

    def uuid7_str() -> str:
        return str(uuid_utils.uuid7())
except ImportError:
    try:
        from uuid7 import uuid7

        def uuid7_str() -> str:
            return str(uuid7())
    except ImportError:
        import uuid

        def uuid7_str() -> str:
            return str(uuid.uuid4())


@dataclass
class WorkerInfo:
    worker_id: str
    config_id: str
    max_capacity: int
    paused: bool = False
    active_jobs: set[str] = field(default_factory=set)
    ws: WebSocket | None = None

    @property
    def available_capacity(self) -> int:
        return max(0, self.max_capacity - len(self.active_jobs))


class Dispatcher:
    """Manages a single worker connection, job dispatch, and result relay."""

    def __init__(self) -> None:
        self.worker: WorkerInfo | None = None
        # job_id -> asyncio.Future (for transient dispatch — held HTTP requests)
        self.pending_transient: dict[str, asyncio.Future[dict]] = {}
        # job_id -> Job (persisted/async jobs, in-memory)
        self.jobs: dict[str, Job] = {}
        # Monotonically increasing fence token
        self._fence_counter = 0

    def _next_fence_token(self) -> int:
        self._fence_counter += 1
        return self._fence_counter

    # -- Worker registry ------------------------------------------------------

    async def register_worker(self, ws: WebSocket, msg: dict) -> WorkerInfo:
        worker_id = msg["worker_id"]
        config_id = msg["config_id"]
        capacity = msg.get("capacity", 1)

        self.worker = WorkerInfo(
            worker_id=worker_id,
            config_id=config_id,
            max_capacity=capacity,
            ws=ws,
        )
        logger.info(
            "Worker %s registered (config_id=%s, capacity=%d)", worker_id, config_id, capacity
        )
        await self._drain_backlog()
        return self.worker

    def remove_worker(self, worker_id: str) -> None:
        if self.worker and self.worker.worker_id == worker_id:
            logger.info("Worker %s disconnected", worker_id)
            self.worker = None

    # -- Transient dispatch (synchronous HTTP → WS → HTTP) --------------------

    async def dispatch_transient(self, payload: dict, timeout: float = 120.0) -> dict:
        """Dispatch a transient job: send to worker via WS, await result.

        Raises RuntimeError on no worker, TimeoutError on timeout.
        """
        worker = self.worker
        if worker is None or worker.paused or worker.ws is None or worker.available_capacity <= 0:
            raise RuntimeError("No worker available")

        job_id = uuid7_str()
        fence_token = self._next_fence_token()

        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict] = loop.create_future()
        self.pending_transient[job_id] = future

        assignment = JobAssignment(
            job_id=job_id,
            mode="transient",
            config_id=worker.config_id,
            payload=payload,
            fence_token=fence_token,
        )

        worker.active_jobs.add(job_id)
        try:
            await worker.ws.send_json(assignment.model_dump())
        except Exception:
            worker.active_jobs.discard(job_id)
            self.pending_transient.pop(job_id, None)
            raise RuntimeError("Failed to send job to worker")

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending_transient.pop(job_id, None)
            worker.active_jobs.discard(job_id)
            raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

        return result

    # -- Persisted job dispatch (async) ----------------------------------------

    async def create_job(self, payload: dict) -> Job:
        job_id = uuid7_str()
        fence_token = self._next_fence_token()
        config_id = self.worker.config_id if self.worker else "default"

        job = Job(
            id=job_id,
            config_id=config_id,
            status=JobStatus.pending,
            payload=payload,
            fence_token=fence_token,
            created_at=time.time(),
        )
        self.jobs[job_id] = job

        # Try to dispatch immediately if a worker is available
        if self.worker and not self.worker.paused and self.worker.available_capacity > 0:
            await self._dispatch_persisted_job(self.worker, job)

        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def _dispatch_persisted_job(self, worker: WorkerInfo, job: Job) -> None:
        assignment = JobAssignment(
            job_id=job.id,
            mode="persisted",
            config_id=worker.config_id,
            payload=job.payload,
            fence_token=job.fence_token,
        )
        try:
            await worker.ws.send_json(assignment.model_dump())
            worker.active_jobs.add(job.id)
            job.status = JobStatus.running
        except Exception:
            logger.exception(
                "Failed to dispatch persisted job %s to worker %s", job.id, worker.worker_id
            )

    # -- Result handling ------------------------------------------------------

    def handle_result(self, job_id: str, result: dict, fence_token: int) -> None:
        # Release worker capacity
        if self.worker:
            self.worker.active_jobs.discard(job_id)

        # Transient path: resolve the held HTTP future
        future = self.pending_transient.pop(job_id, None)
        if future is not None and not future.done():
            future.set_result(result)
            return

        # Persisted path: update in-memory job
        job = self.jobs.get(job_id)
        if job is not None:
            job.status = JobStatus.completed
            job.result = result

    def handle_error(self, job_id: str, error: str, fence_token: int) -> None:
        if self.worker:
            self.worker.active_jobs.discard(job_id)

        # Transient path
        future = self.pending_transient.pop(job_id, None)
        if future is not None and not future.done():
            future.set_result({"_error": error})
            return

        # Persisted path
        job = self.jobs.get(job_id)
        if job is not None:
            job.status = JobStatus.failed
            job.error = error

    def handle_progress(
        self, job_id: str, fence_token: int, progress: float | None, status_message: str | None
    ) -> None:
        # No-op in test server (no real lease), just log
        logger.debug("Progress for job %s: progress=%s msg=%s", job_id, progress, status_message)

    # -- Backlog drain --------------------------------------------------------

    async def _drain_backlog(self) -> None:
        """Dispatch pending persisted jobs to the connected worker."""
        worker = self.worker
        if worker is None:
            return
        for job in list(self.jobs.values()):
            if worker.available_capacity <= 0:
                break
            if job.status == JobStatus.pending:
                await self._dispatch_persisted_job(worker, job)
