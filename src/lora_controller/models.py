from pydantic import BaseModel, Field
from datetime import datetime


class AdapterMeta(BaseModel):
    """Adapter metadata stored in the registry."""

    path: str
    description: str
    rank: int
    base_model: str
    tags: list[str] = []
    passthrough: bool = False
    system_prompt: str = ""
    chat_template_kwargs: dict = {}


class LoadedAdapter(BaseModel):
    """Runtime state of a loaded adapter."""

    name: str
    meta: AdapterMeta
    loaded_at: datetime
    last_used: datetime
    use_count: int = 0


class LoadAdapterRequest(BaseModel):
    lora_name: str
    load_inplace: bool = False


class UnloadAdapterRequest(BaseModel):
    lora_name: str


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stop: list[str] = []
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    raw_completion: bool = Field(
        default=False,
        description=(
            "If True, send `prompt` verbatim to vLLM's /v1/completions with no "
            "chat template — for raw text-continuation tasks (e.g. code fill-in) "
            "where the adapter expects to continue the prompt, not reply to it. "
            "If False (default), wrap `prompt` as a user message through "
            "/v1/chat/completions — required for every adapter here, since the "
            "base model is Instruct-tuned and free-runs into unrelated, often "
            "repetitive text when a raw instruction is fed through the raw "
            "completion endpoint."
        ),
    )


class AdapterStatus(BaseModel):
    """Public-facing adapter info returned by list/get endpoints."""

    name: str
    description: str
    rank: int
    base_model: str
    tags: list[str]
    loaded: bool
    last_used: datetime | None = None
    use_count: int = 0
