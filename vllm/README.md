# Worker vLLM

A Python-based worker container that wraps the vLLM online server Docker image for LLM chat completion inference. This worker connects to queue via WebSocket for push-based job dispatch, supports heartbeat monitoring, and executes actual LLM inference.

## Features

- **vLLM Integration**: Uses vLLM's OpenAI-compatible API server for high-performance LLM inference
- **WebSocket transport**: Connects to queue Durable Object via WebSocket for instant job push
- **Heartbeat monitoring**: Sends periodic heartbeats to the status API
- **Chat Completion Execution**: Processes LLM chat completion jobs with actual inference
- **Lease Renewal**: Automatically renews job leases during long-running inference
- **Graceful shutdown**: Handles SIGTERM/SIGINT with job completion or release
- **Configurable via environment variables**: All parameters are configurable

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
- `VLLM_MODEL` - Model to load in vLLM (e.g., `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mistral-7B-Instruct-v0.2`)

### Optional Environment Variables

#### Worker Configuration

- `CASOLA_API_URL` - API base URL (for heartbeats, without trailing slash)
- `CASOLA_API_TOKEN` - Bearer token for API authentication (for heartbeats)
- `CASOLA_INSTANCE_ID` - Custom instance ID, also used as worker_id (default: `vast-$CONTAINER_ID`)
- `CASOLA_QUEUE_ID` - Queue ID for this worker (default: `vllm-queue`)
- `CASOLA_HEARTBEAT_INTERVAL` - Heartbeat interval in seconds (default: `60`)
- `CASOLA_MAX_JOBS` - Maximum concurrent jobs (default: `1`)
- `CASOLA_LEASE_SECONDS` - Job lease duration in seconds (default: `30`)
- `CASOLA_RENEW_LEASE_AFTER` - Renew lease after this many seconds (default: `80%` of `CASOLA_LEASE_SECONDS`)
- `CASOLA_SHUTDOWN_GRACE_PERIOD` - Time to wait for job completion on shutdown in seconds (default: `5.0`)

#### vLLM Configuration

- `VLLM_HOST` - vLLM server host (default: `127.0.0.1`)
- `VLLM_PORT` - vLLM server port (default: `8000`)
- `VLLM_STARTUP_TIMEOUT` - Time to wait for vLLM server to start in seconds (default: `120`)
- `VLLM_EXTRA_ARGS` - Additional arguments to pass to vLLM server (e.g., `--tensor-parallel-size 2 --gpu-memory-utilization 0.9`)

### Example

```bash
docker run --gpus all \
  -e CASOLA_API_URL=https://api.example.com \
  -e CASOLA_API_TOKEN=your-api-token \
  -e CASOLA_QUEUE_ID=vllm-queue \
  -e VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf \
  -e VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.9" \
  ghcr.io/casola-ai/worker-vllm:latest
```

### Example with Hugging Face Token

For gated models (e.g., Llama 2, Llama 3):

```bash
docker run --gpus all \
  -e CASOLA_API_URL=https://api.example.com \
  -e CASOLA_API_TOKEN=your-api-token \
  -e CASOLA_QUEUE_ID=vllm-queue \
  -e VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf \
  -e HF_TOKEN=your-huggingface-token \
  -e VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.9" \
  ghcr.io/casola-ai/worker-vllm:latest
```

## How It Works

1. **Container Startup** (via `entrypoint.sh`):
   - Starts vLLM server as an independent process
   - Starts worker as an independent process
   - Monitors both processes continuously
   - **If vLLM fails**: Container exits immediately (worker gets grace period to report heartbeat)
   - **If worker exits**: Container exits with worker's exit code
   - Handles SIGTERM/SIGINT with proper cleanup

2. **Worker Initialization**:
   - Validates required environment variables
   - Waits for vLLM server to be ready (up to `VLLM_STARTUP_TIMEOUT` seconds)
   - Initializes OpenAI client pointing to local vLLM server

3. **Heartbeat Loop**:
   - Sends heartbeat with status "running" every `CASOLA_HEARTBEAT_INTERVAL` seconds

4. **Job Dispatch**:
   - Receives jobs pushed via WebSocket from queue

5. **Job Processing**:
   - Processes `chat_completion` or `llm_inference` job types
   - Extracts messages and parameters from job payload
   - Executes chat completion using vLLM's OpenAI-compatible API
   - Automatically renews job lease during long-running inference
   - Completes or fails jobs via WebSocket messages back to queue

6. **Graceful Shutdown**:
   - On SIGTERM/SIGINT, entrypoint script sends SIGTERM to worker
   - Worker sends drain message to queue (stops new job dispatch)
   - Worker waits up to `CASOLA_SHUTDOWN_GRACE_PERIOD` seconds for current job to complete
   - Worker sends final heartbeat with status "stopping"
   - After worker shutdown, entrypoint script stops vLLM server
   - Container exits

## Job Payload Format

Jobs should have the following structure:

```json
{
  "id": "job-123",
  "job_type": "chat_completion",
  "payload": {
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "model": "default",
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": ["\n\n"]
  }
}
```

### Payload Parameters

- `messages` (required): Array of message objects with `role` and `content`
- `model` (optional): Model identifier (default: "default")
- `temperature` (optional): Sampling temperature 0.0-2.0 (default: 0.7)
- `max_tokens` (optional): Maximum tokens to generate (default: 512)
- `top_p` (optional): Nucleus sampling parameter (default: 1.0)
- `frequency_penalty` (optional): Frequency penalty -2.0 to 2.0 (default: 0.0)
- `presence_penalty` (optional): Presence penalty -2.0 to 2.0 (default: 0.0)
- `stop` (optional): Array of stop sequences

## Job Result Format

Successful jobs return results in this format:

```json
{
  "status": "success",
  "job_type": "chat_completion",
  "processed_at": "2024-01-29T12:00:00.000Z",
  "completion": {
    "id": "cmpl-123",
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "The capital of France is Paris."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 25,
      "completion_tokens": 10,
      "total_tokens": 35
    }
  }
}
```

## API Integration

The worker connects to queue via WebSocket for job dispatch and uses the API for heartbeats:

- WebSocket connection to queue for job push, completion, failure, and progress
- `POST /api/heartbeats` - Send heartbeat status updates (via API)

## Supported Models

Any model supported by vLLM can be used. Popular choices include:

- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `google/gemma-7b-it`
- `Qwen/Qwen2-7B-Instruct`

## GPU Requirements

- Requires NVIDIA GPU with CUDA support
- GPU memory requirements depend on model size:
  - 7B models: ~16GB VRAM
  - 13B models: ~26GB VRAM
  - 70B models: ~140GB VRAM (requires multiple GPUs)

## Development

### Local Testing

You can run the worker locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Start vLLM server in one terminal
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model meta-llama/Llama-2-7b-chat-hf

# Run worker in another terminal
export CASOLA_API_URL=https://api.example.com
export CASOLA_API_TOKEN=your-api-token
export CASOLA_QUEUE_ID=vllm-queue
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8000

python worker.py
```

### Testing with Different Models

```bash
# Test with Mistral
docker run --gpus all \
  -e CASOLA_API_URL=https://api.example.com \
  -e CASOLA_API_TOKEN=your-api-token \
  -e VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
  ghcr.io/casola-ai/worker-vllm:latest

# Test with multi-GPU setup
docker run --gpus all \
  -e CASOLA_API_URL=https://api.example.com \
  -e CASOLA_API_TOKEN=your-api-token \
  -e VLLM_MODEL=meta-llama/Llama-2-70b-chat-hf \
  -e VLLM_EXTRA_ARGS="--tensor-parallel-size 4" \
  ghcr.io/casola-ai/worker-vllm:latest
```

## Troubleshooting

### vLLM Server Fails to Start

- Check GPU availability: `nvidia-smi`
- Verify model name is correct
- Ensure sufficient GPU memory
- Check Hugging Face token for gated models

### Worker Can't Connect to vLLM

- Increase `VLLM_STARTUP_TIMEOUT` for large models
- Check vLLM server logs in container output
- Verify `VLLM_HOST` and `VLLM_PORT` settings

### Jobs Timing Out

- Increase `CASOLA_LEASE_SECONDS` for long inference
- Decrease `CASOLA_RENEW_LEASE_AFTER` for more frequent renewals
- Reduce `max_tokens` in job payload

### Out of Memory Errors

- Reduce `--gpu-memory-utilization` (default 0.9)
- Use smaller model
- Reduce batch size with `--max-num-seqs`
