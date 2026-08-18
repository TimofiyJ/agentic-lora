"""LoRA Controller — FastAPI service wrapping vLLM's dynamic LoRA endpoints.

Adds a registry, LRU eviction, and auto-load on inference on top of vLLM's
native load/unload API.

Environment variables:
    VLLM_URL:         vLLM server URL (default: http://localhost:8000)
    ADAPTERS_CONFIG:  Path to adapters.yaml (default: adapters.yaml)
    MAX_LORAS:        Max simultaneous adapters (default: 4)
    VLLM_TIMEOUT:     HTTP timeout in seconds (default: 30)
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from .client import VLLMClient
from .registry import AdapterRegistry
from .scheduler import AdapterScheduler
from .models import AdapterStatus, InferRequest, LoadAdapterRequest


registry: AdapterRegistry
scheduler: AdapterScheduler
vllm: VLLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, scheduler, vllm
    registry = AdapterRegistry(os.getenv("ADAPTERS_CONFIG", "adapters.yaml"))
    scheduler = AdapterScheduler(int(os.getenv("MAX_LORAS", "4")))
    vllm = VLLMClient(os.getenv("VLLM_URL", "http://localhost:8000"), float(os.getenv("VLLM_TIMEOUT", "30")))
    yield


app = FastAPI(
    title="LoRA Controller",
    description="Adapter pool service wrapping vLLM's dynamic LoRA management",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/adapters", response_model=list[AdapterStatus])
def list_adapters():
    """List all registered adapters with their current GPU load status."""
    loaded = scheduler.all_loaded()
    return [
        AdapterStatus(
            name=name,
            description=meta.description,
            rank=meta.rank,
            base_model=meta.base_model,
            tags=meta.tags,
            loaded=name in loaded,
            last_used=loaded[name].last_used if name in loaded else None,
            use_count=loaded[name].use_count if name in loaded else 0,
        )
        for name, meta in registry.all().items()
    ]


@app.get("/adapters/loaded")
def list_loaded():
    """Return adapters currently resident on GPU.

    The planner calls this to know what is available without a load penalty.
    """
    return {
        name: {
            "loaded_at": state.loaded_at.isoformat(),
            "last_used": state.last_used.isoformat(),
            "use_count": state.use_count,
            "rank": state.meta.rank,
            "description": state.meta.description,
        }
        for name, state in scheduler.all_loaded().items()
    }


@app.get("/adapters/{name}", response_model=AdapterStatus)
def get_adapter(name: str):
    """Return status for a single adapter.

    Args:
        name: Adapter name as defined in adapters.yaml.

    Raises:
        HTTPException: 404 if the adapter is not in the registry.
    """
    meta = registry.get(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not in registry")
    state = scheduler.get(name)
    return AdapterStatus(
        name=name,
        description=meta.description,
        rank=meta.rank,
        base_model=meta.base_model,
        tags=meta.tags,
        loaded=scheduler.is_loaded(name),
        last_used=state.last_used if state else None,
        use_count=state.use_count if state else 0,
    )


@app.post("/adapters/{name}/load")
async def load_adapter(name: str, body: LoadAdapterRequest | None = None):
    """Load an adapter into vLLM GPU memory, evicting LRU if at capacity.

    Args:
        name: Adapter name to load.
        body: Optional request body to pass load_inplace flag.

    Raises:
        HTTPException: 404 if the adapter is not in the registry.
    """
    if scheduler.is_loaded(name):
        return {"status": "already_loaded", "adapter": name}

    meta = registry.get(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not in registry")

    if scheduler.at_capacity():
        victim = scheduler.eviction_candidate()
        if victim:
            await _evict(victim)

    await vllm.load_adapter(name, meta, load_inplace=body.load_inplace if body else False)
    scheduler.record_load(name, meta)
    return {"status": "loaded", "adapter": name}


@app.post("/adapters/{name}/unload")
async def unload_adapter(name: str):
    """Unload an adapter from vLLM GPU memory.

    Args:
        name: Adapter name to unload.
    """
    if not scheduler.is_loaded(name):
        return {"status": "not_loaded", "adapter": name}
    await vllm.unload_adapter(name)
    scheduler.record_unload(name)
    return {"status": "unloaded", "adapter": name}


@app.post("/infer/{adapter_name}")
async def infer(adapter_name: str, body: InferRequest):
    """Run inference with a specific adapter, auto-loading it if needed.

    Args:
        adapter_name: Adapter to use. Auto-loaded (with LRU eviction) if not on GPU.
        body: Inference request parameters.

    Raises:
        HTTPException: 404 if the adapter is not in the registry.
    """
    if adapter_name not in registry:
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter_name}' not in registry")

    if not scheduler.is_loaded(adapter_name):
        await load_adapter(adapter_name)

    result = await vllm.completions(
        model=adapter_name,
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stop=body.stop,
    )
    scheduler.touch(adapter_name)
    return result


@app.post("/registry/reload")
def reload_registry():
    """Hot-reload adapters.yaml without restarting the service."""
    registry.reload()
    return {"status": "reloaded", "adapter_count": len(registry.all())}


@app.get("/health")
async def health():
    """Return service health and vLLM reachability."""
    return {
        "status": "ok",
        "vllm_reachable": await vllm.health(),
        "loaded_adapters": len(scheduler.loaded_names()),
        "max_loras": scheduler.max_loras,
    }


async def _evict(name: str) -> None:
    """Unload an adapter from vLLM and remove it from scheduler state.

    Args:
        name: Adapter name to evict.
    """
    await vllm.unload_adapter(name)
    scheduler.record_unload(name)
