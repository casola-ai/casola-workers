# Worker Integration Test

A Python-based worker container for integration testing that connects to queue via WebSocket for push-based job dispatch.

## Features

- **WebSocket transport**: Connects to queue Durable Object via WebSocket for instant job push
- **Heartbeat monitoring**: Sends periodic heartbeats to the status API (optional)
- **Job processing**: Receives, processes, and completes/fails jobs
- **Graceful shutdown**: Drains connections and sends "stopping" status on SIGTERM/SIGINT
- **Configurable via environment variables**: All URLs and parameters are configurable

## Building

```bash
./build.sh
```

Or with custom image name/tag:

```bash
IMAGE_NAME=my-worker IMAGE_TAG=v1.0 ./build.sh
```

## Running

### Required Environment Variables

- `CASOLA_WS_URL` - WebSocket URL for queue connection (e.g., `wss://us.casola-staging.net`)
- `CASOLA_CONFIG_ID` - Config ID for this worker (set by scheduler)

### Optional Environment Variables

- `CASOLA_API_URL` - API base URL (for heartbeats, without trailing slash)
- `CASOLA_API_TOKEN` - Bearer token for API authentication (for heartbeats)
- `CASOLA_INSTANCE_ID` - Custom instance ID, also used as worker_id (default: `vast-$CONTAINER_ID` or `vast-<pid>` if CONTAINER_ID not set)
- `CASOLA_QUEUE_ID` - Queue ID for this worker (default: `test-queue`)
- `CASOLA_HEARTBEAT_INTERVAL` - Heartbeat interval in seconds (default: `60`)
- `CASOLA_MAX_JOBS` - Maximum concurrent jobs (default: `1`)
- `CASOLA_CAPACITY` - Worker capacity advertised to queue (default: `1`)
- `CASOLA_LEASE_SECONDS` - Job lease duration in seconds (default: `30`)
- `CASOLA_RENEW_LEASE_AFTER` - Renew lease after this many seconds (default: `80%` of `CASOLA_LEASE_SECONDS`)
- `CASOLA_SIMULATED_EXECUTION_TIME` - Simulated job execution time in seconds (default: `2.0`)
- `CASOLA_SIMULATED_ERROR_RATE` - Probability of simulated job failure, 0.0-1.0 (default: `0.0`)
- `CASOLA_SHUTDOWN_GRACE_PERIOD` - Time to wait for job completion on shutdown in seconds (default: `5.0`)

### Example

```bash
docker run \
  -e CASOLA_WS_URL=wss://us.casola-staging.net \
  -e CASOLA_INSTANCE_ID=test-worker-1 \
  -e CASOLA_QUEUE_ID=test-queue \
  -e CASOLA_SIMULATED_EXECUTION_TIME=3.0 \
  -e CASOLA_SIMULATED_ERROR_RATE=0.1 \
  ghcr.io/casola-ai/worker-integration-test:latest
```

## How It Works

1. **Initialization**: Validates required environment variables and sets up signal handlers
2. **WebSocket Connection**: Connects to queue and registers with capabilities (job types, capacity)
3. **Heartbeat Loop**: Optionally sends heartbeat with status "running" every `CASOLA_HEARTBEAT_INTERVAL` seconds (requires API URL/token)
4. **Job Processing**:
   - Receives jobs pushed via WebSocket from queue
   - Simulates work with configurable execution time (`CASOLA_SIMULATED_EXECUTION_TIME`)
   - Randomly fails jobs based on error rate (`CASOLA_SIMULATED_ERROR_RATE`)
   - Completes or fails jobs via WebSocket messages back to queue
5. **Graceful Shutdown**:
   - On SIGTERM/SIGINT, sends drain message to queue (stops new job dispatch)
   - Waits up to `CASOLA_SHUTDOWN_GRACE_PERIOD` seconds for current job to complete
   - Sends final heartbeat with status "stopping"

## Testing Graceful Shutdown

```bash
# In one terminal, start the container
docker run --name test-worker \
  -e CASOLA_WS_URL=wss://us.casola-staging.net \
  -e CASOLA_QUEUE_ID=test-queue \
  ghcr.io/casola-ai/worker-integration-test:latest

# In another terminal, send SIGTERM
docker stop test-worker
```

The container will send a final heartbeat with status "stopping" before exiting.

## Development

### Local Testing

You can run the worker locally without Docker:

```bash
pip install -r requirements.txt

export CASOLA_WS_URL=ws://localhost:8788

python worker.py
```

### Customizing Job Processing

The `process_job` method in `worker.py` contains the job processing logic. Modify this method to implement custom job processing behavior:

```python
def process_job(self, job: Dict[str, Any]) -> bool:
    job_id = job.get('id')
    job_type = job.get('job_type')
    payload = job.get('payload', {})

    # Your custom processing logic here

    result = {
        "status": "success",
        # Your result data
    }

    return self.transport.complete_job(job_id, result)
```
