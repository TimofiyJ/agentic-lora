"""Async HTTP client for vLLM's LoRA management and inference endpoints."""

import httpx
from .models import AdapterMeta


class VLLMClient:
    """Thin async wrapper around vLLM's REST API.

    Args:
        base_url: Base URL of the vLLM server, e.g. ``http://localhost:8000``.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    async def load_adapter(self, name: str, meta: AdapterMeta, load_inplace: bool = False) -> None:
        """Send a load request to vLLM.

        Args:
            name: Adapter name vLLM will use to identify it in inference requests.
            meta: Registry metadata containing the adapter path.
            load_inplace: If True, hot-swap weights without unloading first.

        Raises:
            httpx.HTTPStatusError: If vLLM returns a non-2xx response.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": meta.path, "load_inplace": load_inplace},
            )
            # vLLM returns 400 "already loaded" if the adapter survived a controller restart; treat as success
            if resp.status_code == 400 and "already been loaded" in resp.text:
                return
            resp.raise_for_status()

    async def unload_adapter(self, name: str) -> None:
        """Send an unload request to vLLM.

        Args:
            name: Adapter name to remove from GPU memory.

        Raises:
            httpx.HTTPStatusError: If vLLM returns a non-2xx response.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/v1/unload_lora_adapter",
                json={"lora_name": name},
            )
            resp.raise_for_status()

    async def completions(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> dict:
        """Call vLLM's /v1/completions endpoint.

        Args:
            model: Adapter name to use (vLLM routes to the loaded LoRA by name).
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Optional stop sequences.

        Returns:
            Raw vLLM completions response dict.

        Raises:
            httpx.HTTPStatusError: If vLLM returns a non-2xx response.
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base}/v1/completions", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def health(self) -> bool:
        """Return True if the vLLM server is reachable and healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base}/health")
                return resp.status_code == 200
        except Exception:
            return False
