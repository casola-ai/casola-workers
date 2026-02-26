# Casola Test Server

Standalone test server for 3rd-party GPU worker integration testing. Simulates both the Queue DO's WebSocket server and the API's OpenAI-compatible REST surface, so developers can test their workers locally without the full Cloudflare stack.

```
Client (curl/SDK) ──REST──► Test Server ──WS──► Worker (your engine)
                   ◄─────── Test Server ◄─────── Worker
```

## Quick Start

```bash
# From the repo root — install the test server
cd workers/test-server
pip install -e ".[dev]"

# Install worker SDK + integration-test deps
cd ..
pip install -e .
pip install -r integration-test/requirements.txt

# Terminal 1 — start the test server
python -m casola_test_server.app

# Terminal 2 — start the integration-test worker
CASOLA_WS_URL=ws://localhost:8788 \
CASOLA_CONFIG_ID=default \
CASOLA_INSTANCE_ID=local-worker \
CASOLA_SIMULATED_EXECUTION_TIME=0.1 \
python workers/integration-test/worker.py

# Terminal 3 — send a chat completion request
curl -s http://localhost:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"hello"}]}' | python -m json.tool

# Send an embeddings request
curl -s http://localhost:8788/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","input":"hello world"}' | python -m json.tool

# Check server health
curl -s http://localhost:8788/health | python -m json.tool
```

The integration-test worker handles all standard tasks (`openai/chat-completion`, `openai/embeddings`, `openai/audio-speech`, `openai/audio-transcription`, `fal/text-to-image`, etc.) and returns mock responses, so the full request→dispatch→result flow works end-to-end.

## Endpoints

### OpenAI-compatible (transient dispatch)

| Method | Path | Task |
|--------|------|------|
| POST | `/v1/chat/completions` | `openai/chat-completion` |
| POST | `/v1/embeddings` | `openai/embeddings` |
| POST | `/v1/audio/speech` | `openai/audio-speech` |
| POST | `/v1/audio/transcriptions` | `openai/audio-transcription` |
| POST | `/v1/images/generations` | `fal/text-to-image` |
| GET | `/v1/models` | List connected worker as a model |

### Job routes (persisted/async)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs` | Create a persisted job |
| GET | `/jobs/{job_id}` | Poll job status and result |

### WebSocket

| Path | Description |
|------|-------------|
| `/ws` | Worker connection endpoint |

### Utility

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health with worker/job counts |

## Running Tests

```bash
cd workers/test-server
pip install -e ".[dev]"
pytest
```

## Design Decisions

- **Single worker** — one worker connection at a time. A new connection replaces the previous one.
- **In-memory only** — no persistence, restarts clean. Intentional for simplicity.
- **No auth** — workers connect freely. No tokens needed for local testing.
- **No R2/blob offload** — all payloads inline. Test payloads are small.
- **No streaming/SSE** — matches production behavior.
- **Port 8788** — matches existing `CASOLA_WS_URL=ws://localhost:8788` convention.
