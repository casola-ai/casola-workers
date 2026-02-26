# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python SDK and reference Docker-based GPU workers for the [Casola](https://casola.ai) inference platform. Workers connect to a Casola queue via WebSocket to receive LLM/multimodal inference jobs and report results.

## Build Commands

```bash
# Install SDK locally
pip install -e .

# Build Docker images (context is always repo root)
docker build -t casola-vllm -f vllm/Dockerfile .
docker build -t casola-vllm-omni -f vllm-omni/Dockerfile .

# Build + push to GHCR (uses build.sh scripts)
cd vllm && bash build.sh        # pushes ghcr.io/casola-ai/worker-vllm
cd vllm-omni && bash build.sh   # pushes ghcr.io/casola-ai/worker-vllm-omni
```

There is no test suite, linter, or CI configuration in this repository.

## Architecture

### SDK (`casola_worker/`)

The installable package (`casola-workers`) provides three modules:

- **`transport.py`** — `QueueTransport` ABC with `WebSocketTransport` implementation. Handles WebSocket connection to `{ws_url}/ws`, reconnection with exponential backoff (1s→60s, 30% jitter), job dispatch in daemon threads, lease renewal every 30s for persisted jobs, and large payload offloading to R2 via presigned URLs (>256KB threshold). Factory: `create_transport(config)`.
- **`system_metrics.py`** — `SystemMetricsReporter` background thread that collects GPU metrics via `nvidia-smi` and POSTs batches to a metrics endpoint.
- **`gpu_health.py`** — `run_gpu_health_checks(expected_gpu_name, expected_vram_gb)` validates GPU name, VRAM, idle state, ECC errors, and power limits before worker startup.

### vLLM Worker (`vllm/worker.py`)

~1300-line `VLLMWorker` class that orchestrates: GPU health checks → spawn vLLM subprocess → poll `/health` until ready → connect WebSocket transport → process jobs. Supports multiple task types routed via `payload.task`:

- `openai/chat-completion` — `/v1/chat/completions`
- `openai/embeddings` — `/v1/embeddings`
- `openai/audio-speech` — TTS
- `openai/audio-transcription` — STT
- `fal/text-to-video`, `fal/image-to-video`, `fal/speech-to-video` — video generation

Includes backpressure monitoring (polls vLLM Prometheus `/metrics` for `vllm:num_requests_waiting`), log buffering/shipping, heartbeat thread, and graceful SIGTERM/SIGINT shutdown with drain.

### vLLM-Omni Worker (`vllm-omni/`)

Reuses `worker.py` and `entrypoint.sh` from `vllm/`. Only differs in Dockerfile base image (`vllm/vllm-omni:v0.14.0`) and sets `VLLM_SERVE_COMMAND=vllm-omni`.

## Key Conventions

- **Python 3.10+** syntax: `str | None`, `list[str]` generics, walrus operator
- **Thread safety**: all shared state (`_ws`, `_active_persisted`, `current_job_id`, `log_buffer`) guarded by `threading.Lock`
- **All background threads are daemon threads** for clean process exit
- **Docker build context is the repo root**, not subdirectories — Dockerfiles `COPY casola_worker/` and `COPY vllm/` as separate layers
- **Observability is best-effort**: heartbeat, metrics, and log shipping failures are logged but never crash the worker
- **WebSocket protocol**: client sends `register`, `job_result`/`job_result_ref`, `job_error`, `job_progress`, `drain`/`pause`/`resume`; server sends `job`, `job_revoked`, `error`
- **Job modes**: `transient` (fire-and-forget) vs `persisted` (tracked with fence tokens, auto lease renewal)
- **vLLM error classification**: regex patterns on stdout categorize errors as fatal (host marked bad) or retryable (OOM, segfault, network)
