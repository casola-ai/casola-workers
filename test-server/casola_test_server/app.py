"""FastAPI application: WebSocket endpoint, OpenAI REST routes, job routes, health."""

from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from .dispatch import Dispatcher
from .models import (
    AudioSpeechRequest,
    ChatCompletionRequest,
    EmbeddingsRequest,
    ImageGenerationRequest,
    JobStatus,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="Casola Test Server")
    dispatcher = Dispatcher()

    # -------------------------------------------------------------------------
    # WebSocket endpoint
    # -------------------------------------------------------------------------

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        worker_id: str | None = None

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "register":
                    worker = await dispatcher.register_worker(ws, msg)
                    worker_id = worker.worker_id

                elif msg_type == "job_result":
                    dispatcher.handle_result(
                        msg["job_id"], msg.get("result", {}), msg.get("fence_token", 0)
                    )

                elif msg_type == "job_error":
                    dispatcher.handle_error(
                        msg["job_id"], msg.get("error", "Unknown error"), msg.get("fence_token", 0)
                    )

                elif msg_type == "job_progress":
                    dispatcher.handle_progress(
                        msg["job_id"],
                        msg.get("fence_token", 0),
                        msg.get("progress"),
                        msg.get("status_message"),
                    )

                elif msg_type == "drain":
                    if dispatcher.worker:
                        dispatcher.worker.paused = True
                        dispatcher.worker.max_capacity = 0
                        logger.info("Worker %s drained", worker_id)

                elif msg_type == "pause":
                    if dispatcher.worker:
                        dispatcher.worker.paused = True
                        logger.info("Worker %s paused", worker_id)

                elif msg_type == "resume":
                    if dispatcher.worker:
                        dispatcher.worker.paused = False
                        logger.info("Worker %s resumed", worker_id)
                        await dispatcher._drain_backlog()

                else:
                    await ws.send_json(
                        {"type": "error", "message": f"Unknown message type: {msg_type}"}
                    )

        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("WebSocket error for worker %s", worker_id)
        finally:
            if worker_id:
                dispatcher.remove_worker(worker_id)

    # -------------------------------------------------------------------------
    # OpenAI-compatible REST routes
    # -------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        payload = {
            "task": "openai/chat-completion",
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.stop is not None:
            payload["stop"] = req.stop

        try:
            result = await dispatcher.dispatch_transient(payload)
        except RuntimeError:
            raise HTTPException(503, detail="No worker available")
        except TimeoutError as e:
            raise HTTPException(504, detail=str(e))

        if "_error" in result:
            raise HTTPException(502, detail=result["_error"])

        # Return the completion object directly if present
        completion = result.get("completion")
        if completion:
            return completion

        # Fallback: wrap raw result
        return result

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingsRequest):
        payload: dict = {
            "task": "openai/embeddings",
            "model": req.model,
            "input": req.input,
        }
        if req.encoding_format is not None:
            payload["encoding_format"] = req.encoding_format
        if req.dimensions is not None:
            payload["dimensions"] = req.dimensions

        try:
            result = await dispatcher.dispatch_transient(payload)
        except RuntimeError:
            raise HTTPException(503, detail="No worker available")
        except TimeoutError as e:
            raise HTTPException(504, detail=str(e))

        if "_error" in result:
            raise HTTPException(502, detail=result["_error"])

        embeddings_data = result.get("embeddings", [])
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": e.get("index", i),
                    "embedding": e.get("embedding", []),
                }
                for i, e in enumerate(embeddings_data)
            ],
            "model": req.model,
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
            },
        }

    @app.post("/v1/audio/speech")
    async def audio_speech(req: AudioSpeechRequest):
        payload: dict = {
            "task": "openai/audio-speech",
            "model": req.model,
            "input": req.input,
            "voice": req.voice,
        }
        if req.response_format is not None:
            payload["response_format"] = req.response_format
        if req.speed is not None:
            payload["speed"] = req.speed

        try:
            result = await dispatcher.dispatch_transient(payload)
        except RuntimeError:
            raise HTTPException(503, detail="No worker available")
        except TimeoutError as e:
            raise HTTPException(504, detail=str(e))

        if "_error" in result:
            raise HTTPException(502, detail=result["_error"])

        # Return audio binary if base64-encoded audio is in result
        import base64

        from starlette.responses import Response

        audio_b64 = result.get("audio_base64")
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            fmt = result.get("format", req.response_format or "mp3")
            content_type = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
                "wav": "audio/wav",
            }.get(fmt, "application/octet-stream")
            return Response(content=audio_bytes, media_type=content_type)

        return result

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request):
        # Transcription uses multipart form, so we parse manually
        form = await request.form()
        model = str(form.get("model", "whisper-1"))
        language = str(form.get("language", "")) or None
        response_format = str(form.get("response_format", "json"))

        payload: dict = {
            "task": "openai/audio-transcription",
            "model": model,
            "response_format": response_format,
        }
        if language:
            payload["language"] = language

        # Read audio file if provided
        audio_file = form.get("file")
        if audio_file and hasattr(audio_file, "read"):
            import base64

            audio_bytes = await audio_file.read()
            payload["audio_base64"] = base64.b64encode(audio_bytes).decode()

        try:
            result = await dispatcher.dispatch_transient(payload)
        except RuntimeError:
            raise HTTPException(503, detail="No worker available")
        except TimeoutError as e:
            raise HTTPException(504, detail=str(e))

        if "_error" in result:
            raise HTTPException(502, detail=result["_error"])

        text = result.get("text", "")
        if response_format == "text":
            from starlette.responses import PlainTextResponse

            return PlainTextResponse(text)

        return {"text": text}

    @app.post("/v1/images/generations")
    async def image_generations(req: ImageGenerationRequest):
        model = req.model or "dall-e-3"

        # Parse size
        width, height = 1024, 1024
        if req.size:
            parts = req.size.split("x")
            if len(parts) == 2:
                width, height = int(parts[0]), int(parts[1])

        payload: dict = {
            "task": "fal/text-to-image",
            "model": model,
            "prompt": req.prompt,
            "num_images": req.n or 1,
            "image_size": {"width": width, "height": height},
        }

        try:
            result = await dispatcher.dispatch_transient(payload)
        except RuntimeError:
            raise HTTPException(503, detail="No worker available")
        except TimeoutError as e:
            raise HTTPException(504, detail=str(e))

        if "_error" in result:
            raise HTTPException(502, detail=result["_error"])

        images = result.get("images", [])
        return {
            "created": int(time.time()),
            "data": [{"url": img.get("url", ""), "revised_prompt": req.prompt} for img in images],
        }

    @app.get("/v1/models")
    async def list_models():
        """Return the connected worker's config_id as a model."""
        models = []
        if dispatcher.worker:
            models.append(
                {
                    "id": dispatcher.worker.config_id,
                    "object": "model",
                    "owned_by": "casola-test-server",
                }
            )
        return {"object": "list", "data": models}

    # -------------------------------------------------------------------------
    # Job routes (persisted/async)
    # -------------------------------------------------------------------------

    @app.post("/jobs")
    async def create_job(request: Request):
        body = await request.json()
        payload = body.get("payload", {})
        job = await dispatcher.create_job(payload)
        return job.model_dump()

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        job = dispatcher.get_job(job_id)
        if job is None:
            raise HTTPException(404, detail="Job not found")
        return job.model_dump()

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    @app.get("/health")
    async def health():
        pending = sum(1 for j in dispatcher.jobs.values() if j.status == JobStatus.pending)
        return {
            "status": "ok",
            "worker_connected": dispatcher.worker is not None,
            "pending_jobs": pending,
        }

    return app


def main() -> None:
    """Run the test server."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8788)


if __name__ == "__main__":
    main()
