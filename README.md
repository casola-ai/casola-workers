# Casola Workers

SDK and reference workers for building GPU inference containers on the [Casola](https://casola.ai) platform.

## What's included

- **`vllm/`** — Reference vLLM worker (standard LLM inference)
- **`vllm-omni/`** — Reference vLLM-Omni worker (multi-modal: TTS, image, video)
- **`your-worker-name/`** — Minimal skeleton for building your own worker (start here)
- **`test-server/`** — Local test server that simulates the Casola queue and REST API
- **`integration-test/`** — Mock worker for end-to-end testing

## Quick start (local testing)

Run a full local loop — test server, mock worker, and a request — with no cloud dependencies:

```bash
# Install the test server and mock worker
pip install -e "test-server/.[dev]"
pip install -r integration-test/requirements.txt

# Terminal 1 — start the test server
python -m casola_test_server.app

# Terminal 2 — start the mock worker
CASOLA_WS_URL=ws://localhost:8788 \
CASOLA_CONFIG_ID=default \
CASOLA_INSTANCE_ID=local-worker \
CASOLA_SIMULATED_EXECUTION_TIME=0.1 \
python integration-test/worker.py

# Terminal 3 — send a request
curl -s http://localhost:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"hello"}]}' | python -m json.tool
```

See [`test-server/README.md`](test-server/README.md) for all available endpoints.

## Deploying to Casola

Build and run the vLLM worker with a GPU:

```bash
docker build -t casola-vllm -f vllm/Dockerfile .
docker run --gpus all \
  -e CASOLA_API_URL=https://api.casola.ai \
  -e CASOLA_API_TOKEN=<your-token> \
  -e VLLM_MODEL=<model-id> \
  casola-vllm
```

See [`vllm/README.md`](vllm/README.md) for detailed configuration and environment variables.

## Building custom workers

Install the SDK:

```bash
pip install casola-workers
```

Use `casola_worker.transport` to connect to the Casola queue and receive jobs:

```python
from casola_worker.transport import create_transport

config = {
    "api_url": "https://api.casola.ai",
    "api_token": "your-token",
    "queue_id": "your-queue",
    "config_id": "your-config",
}

transport = create_transport(config)
transport.on_job = my_process_function
transport.start()  # blocking
```

Test your worker locally by pointing it at the test server:

```bash
CASOLA_WS_URL=ws://localhost:8788 \
CASOLA_CONFIG_ID=default \
python your_worker.py
```

## Project structure

```
your-worker-name/     # Minimal worker skeleton (start here)
  worker.py
  README.md
vllm/                 # Standard vLLM worker
  Dockerfile
  worker.py
  entrypoint.sh
vllm-omni/            # Multi-modal vLLM worker
  Dockerfile
test-server/          # Local test server (simulates queue + REST API)
  casola_test_server/
  tests/
integration-test/     # Mock worker for E2E testing
  worker.py
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
