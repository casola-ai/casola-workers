"""Pydantic models for WebSocket messages, OpenAI requests/responses, and jobs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

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
    progress: Optional[float] = None
    status_message: Optional[str] = None


class JobRevoked(BaseModel):
    type: str = "job_revoked"
    job_id: str
    reason: str


# =============================================================================
# OpenAI request models (subset matching what workers actually receive)
# =============================================================================


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[list[str] | str] = None


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None


class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"


# =============================================================================
# Job model
# =============================================================================


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    id: str
    config_id: str
    status: JobStatus = JobStatus.pending
    payload: Any = {}
    result: Optional[Any] = None
    error: Optional[str] = None
    fence_token: int = 0
    created_at: float = 0.0
