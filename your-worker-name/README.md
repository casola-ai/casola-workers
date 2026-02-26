# your-worker-name

Minimal skeleton for building a custom Casola inference worker. Copy this directory, rename it, and replace the `process_job` function with your own logic.

## Quick start (local)

```bash
# From workers/ — install the SDK
pip install -e .

# Start the test server (simulates the Casola queue)
pip install -e "test-server/.[dev]"
python -m casola_test_server.app

# In another terminal — run your worker
CASOLA_WS_URL=ws://localhost:8788 \
CASOLA_CONFIG_ID=default \
python your-worker-name/worker.py
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `CASOLA_WS_URL` | yes | WebSocket URL for the queue |
| `CASOLA_CONFIG_ID` | yes | Job config ID this worker serves |
| `CASOLA_INSTANCE_ID` | no | Worker identifier (defaults to PID) |
| `CASOLA_CAPACITY` | no | Concurrent job slots (default `1`) |
| `CASOLA_API_TOKEN` | no | Auth token (required in production) |
| `CASOLA_API_URL` | no | API base URL — enables heartbeats |
| `CASOLA_HEARTBEAT_INTERVAL` | no | Seconds between heartbeats (default `60`) |

## How it works

1. The worker connects to the queue via WebSocket and registers with its `config_id` and `capacity`.
2. The queue dispatches jobs as JSON messages. Each job has an `id`, `payload`, and `mode`.
3. Your `process_job` function runs in its own thread. Use the `transport` to report results:
   - `transport.complete_job(job_id, result_dict)` — success
   - `transport.fail_job(job_id, "error message")` — failure
   - `transport.send_progress(job_id, progress=0.5)` — progress update / lease renewal
4. On shutdown (SIGTERM/SIGINT), the worker drains (stops accepting new jobs) and exits.

### What the transport handles for you

- **WebSocket lifecycle** — connect, register, reconnect with exponential backoff
- **All WS message types** — `job`, `job_revoked`, `job_result_ref`, `error` (incoming); `register`, `job_result`, `job_error`, `job_progress`, `drain`, `pause`/`resume` (outgoing)
- **Lease renewal** — automatically sends `job_progress` every 30 s for persisted jobs so leases don't expire while your code is running
- **Heartbeats** are a separate worker-level liveness signal sent via HTTP to the API (see env vars above)

## Deploying to production

Build a Docker image with your dependencies and set the environment variables above. See the [vLLM worker](../vllm/) for a full production example with health checks and Dockerfile.
