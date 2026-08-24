"""System-aware scheduler: prefetch and unload LoRA adapters via the controller."""

import asyncio
import time
import httpx
from datetime import datetime, UTC

from .models import SchedulingEvent


class ContinuousScheduler:
    """Prefetches and unloads LoRA adapters via the controller API.

    Args:
        controller_url: Base URL of the LoRA controller.
        timeout: HTTP timeout in seconds for controller calls.
    """

    def __init__(self, controller_url: str, timeout: float = 30.0):
        self._controller_url = controller_url.rstrip("/")
        self._timeout = timeout

    async def get_loaded(self) -> set[str]:
        """Return the set of adapter names currently resident in GPU memory.

        Returns:
            Set of adapter name strings reported by the controller.
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._controller_url}/adapters/loaded")
            resp.raise_for_status()
            return set(resp.json().keys())

    async def prefetch(self, adapters: list[str]) -> list[SchedulingEvent]:
        """Load adapters concurrently ahead of their scheduled use.

        Args:
            adapters: Adapter names to prefetch.

        Returns:
            One SchedulingEvent per adapter with event "prefetch" or "prefetch_error".
        """
        async def _load_one(name: str) -> SchedulingEvent:
            t0 = time.monotonic()
            event: str
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(f"{self._controller_url}/adapters/{name}/load")
                    resp.raise_for_status()
                event = "prefetch"
            except Exception:
                event = "prefetch_error"
            return SchedulingEvent(
                event=event,
                adapter=name,
                timestamp=datetime.now(UTC),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        return list(await asyncio.gather(*[_load_one(a) for a in adapters]))

    async def unload_all(self, adapters: set[str]) -> list[SchedulingEvent]:
        """Unload adapters concurrently after a task completes.

        Args:
            adapters: Adapter names to unload.

        Returns:
            One SchedulingEvent per adapter with event "unload".
        """
        async def _unload_one(name: str) -> SchedulingEvent:
            t0 = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(f"{self._controller_url}/adapters/{name}/unload")
                    resp.raise_for_status()
            except Exception:
                pass
            return SchedulingEvent(
                event="unload",
                adapter=name,
                timestamp=datetime.now(UTC),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        return list(await asyncio.gather(*[_unload_one(a) for a in adapters]))
