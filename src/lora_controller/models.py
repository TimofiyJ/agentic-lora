from pydantic import BaseModel, Field
from datetime import datetime


class AdapterMeta(BaseModel):
    """Adapter metadata stored in the registry."""

    path: str
    description: str
    rank: int
    base_model: str
    tags: list[str] = []


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
