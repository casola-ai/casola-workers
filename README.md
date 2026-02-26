# Casola Workers

SDK and reference workers for building GPU inference containers on the [Casola](https://casola.ai) platform.

## What's included

- **`casola_worker/`** — Python SDK providing WebSocket transport, system metrics, and GPU health checks
- **`vllm/`** — Reference vLLM worker (standard LLM inference)
- **`vllm-omni/`** — Reference vLLM-Omni worker (multi-modal: TTS, image, video)

## Quick start

Build and run the vLLM worker:

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

## Project structure

```
casola_worker/        # SDK package
  transport.py        # WebSocket transport (queue connection, job dispatch)
  system_metrics.py   # GPU/CPU/memory metrics reporter
  gpu_health.py       # Pre-startup GPU health checks
vllm/                 # Standard vLLM worker
  Dockerfile
  worker.py
  entrypoint.sh
vllm-omni/            # Multi-modal vLLM worker
  Dockerfile
pyproject.toml        # Package metadata
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
