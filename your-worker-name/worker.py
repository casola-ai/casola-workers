#!/usr/bin/env python3
"""
Casola worker template — minimal skeleton for building a custom inference worker.

Replace the body of `process_job` with your own inference logic.
"""

import logging
import os
import signal
import sys
import threading
from typing import Any, Dict

import requests

# Allow importing casola_worker from parent directory (local dev)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from casola_worker.transport import QueueTransport, create_transport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("your-worker-name")


# -- Configuration -----------------------------------------------------------

WS_URL = os.environ["CASOLA_WS_URL"]  # WebSocket URL for the queue
CONFIG_ID = os.environ["CASOLA_CONFIG_ID"]  # Which job config this worker serves
WORKER_ID = os.environ.get("CASOLA_INSTANCE_ID", f"worker-{os.getpid()}")
CAPACITY = int(os.environ.get("CASOLA_CAPACITY", "1"))  # Concurrent job slots
API_TOKEN = os.environ.get("CASOLA_API_TOKEN")  # Auth token (required in production)
API_URL = os.environ.get("CASOLA_API_URL")  # API base URL (for heartbeats)
HEARTBEAT_INTERVAL = int(os.environ.get("CASOLA_HEARTBEAT_INTERVAL", "60"))


# -- Heartbeats ---------------------------------------------------------------
# Heartbeats tell the control plane this worker is alive. The transport handles
# job-level lease renewal automatically (every 30 s for persisted jobs), but
# heartbeats are a separate, worker-level liveness signal.


def heartbeat_loop(running: threading.Event) -> None:
    """Send periodic heartbeats to the API while `running` is set."""
    if not API_URL or not API_TOKEN:
        return  # heartbeats are optional without API credentials
    url = f"{API_URL}/api/heartbeats"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"}
    payload = {"instance_id": WORKER_ID, "status": "running"}

    while running.is_set():
        try:
            requests.post(url, json=payload, headers=headers, timeout=10)
        except Exception:
            logger.warning("Heartbeat failed", exc_info=True)
        running.wait(timeout=HEARTBEAT_INTERVAL)


# -- Job handler --------------------------------------------------------------


def process_job(job: Dict[str, Any], transport: QueueTransport) -> None:
    """
    Called for each dispatched job. Runs in its own thread.

    `job` dict fields:
        id         – unique job identifier
        config_id  – the config this job was routed to
        payload    – arbitrary dict from the caller (model inputs, params, etc.)
        mode       – "transient" (fire-and-forget) or "persisted" (result stored)
        fence_token – monotonic token for persisted-job lease correctness

    Use `transport` to report results back to the queue:
        transport.complete_job(job_id, result_dict)   – report success
        transport.fail_job(job_id, "error message")   – report failure
        transport.send_progress(job_id, progress=0.5) – progress / lease renewal

    Lease renewal: The transport automatically renews leases for persisted jobs
    every 30 s. For long-running jobs you can also call `send_progress` manually
    to push more granular updates to the caller.
    """
    job_id = job["id"]
    payload = job.get("payload", {})

    logger.info("Received job %s — payload keys: %s", job_id, list(payload.keys()))

    try:
        # TODO: Replace with your inference logic.
        #
        # For long-running work, report progress so callers can display it:
        #   transport.send_progress(job_id, progress=0.25, status_message="loading model")
        #   ...
        #   transport.send_progress(job_id, progress=0.75, status_message="generating")

        result = {"output": "hello from your-worker-name"}

        transport.complete_job(job_id, result)
        logger.info("Completed job %s", job_id)

    except Exception as e:
        logger.exception("Failed job %s", job_id)
        transport.fail_job(job_id, str(e))


# -- Entrypoint ---------------------------------------------------------------


def main() -> None:
    transport = create_transport(
        {
            "ws_url": WS_URL,
            "worker_id": WORKER_ID,
            "config_id": CONFIG_ID,
            "capacity": CAPACITY,
            "api_token": API_TOKEN,
        }
    )

    # Wire up the job handler (transport calls this in a new thread per job)
    transport.on_job = lambda job: process_job(job, transport)

    # Heartbeat thread — worker-level liveness signal to the control plane.
    # (Job-level lease renewal is handled automatically by the transport.)
    running = threading.Event()
    running.set()
    hb_thread = threading.Thread(target=heartbeat_loop, args=(running,), daemon=True)
    hb_thread.start()

    # Graceful shutdown on SIGTERM / SIGINT
    def shutdown(signum, frame):
        logger.info("Shutting down (signal %s)...", signum)
        running.clear()  # stop heartbeats
        transport.drain()  # stop accepting new jobs
        transport.stop()  # close the WebSocket

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info(
        "Starting worker %s (config_id=%s, capacity=%d)",
        WORKER_ID,
        CONFIG_ID,
        CAPACITY,
    )

    # Blocks until stop() is called. Handles all WS messages internally:
    #   incoming: job, job_revoked, job_result_ref (ack), error
    #   outgoing: register, job_result, job_error, job_progress, drain, pause/resume
    # Reconnects automatically with exponential backoff on disconnect.
    transport.start()


if __name__ == "__main__":
    main()
