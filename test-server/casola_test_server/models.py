"""Pydantic models for WebSocket messages, OpenAI requests/responses, and jobs."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

# =============================================================================
# WebSocket protocol models (mirrors queue-types.ts)
# =============================================================================


class RegisterMessage(BaseModel):
    type: str = "register"
    worker_id: str
    config_id: str
    capacity: int = 1


class JobAssignment(BaseModel):
    type: str = "job"
    job_id: str
    mode: str = "transient"  # "transient" | "persisted"
    config_id: str
    payload: Any = {}
    fence_token: int = 0


class JobResultMessage(BaseModel):
    type: str = "job_result"
    job_id: str
    result: Any
    fence_token: int = 0


class JobErrorMessage(BaseModel):
    type: str = "job_error"
    job_id: str
    error: str
    fence_token: int = 0


class JobProgressMessage(BaseModel):
    type: str = "job_progress"
    job_id: str
    fence_token: int = 0
    progress: float | None = None
    status_message: str | None = None


class JobRevoked(BaseModel):
    type: str = "job_revoked"
    job_id: str
    reason: str


# =============================================================================
# OpenAI request models (subset matching what workers actually receive)
# =============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = None
    dimensions: int | None = None


class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: str | None = "mp3"
    speed: float | None = 1.0


class ImageGenerationRequest(BaseModel):
    model: str | None = None
    prompt: str
    n: int | None = 1
    size: str | None = "1024x1024"
    response_format: str | None = "url"


# =============================================================================
# Job model
# =============================================================================


class JobStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    id: str
    config_id: str
    status: JobStatus = JobStatus.pending
    payload: Any = {}
    result: Any | None = None
    error: str | None = None
    fence_token: int = 0
    created_at: float = 0.0
