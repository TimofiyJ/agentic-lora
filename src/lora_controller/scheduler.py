"""Adapter lifecycle manager with LRU eviction."""

from datetime import datetime, UTC
from .models import AdapterMeta, LoadedAdapter


class AdapterScheduler:
    """Tracks adapters resident on GPU and enforces the max_loras capacity limit.

    Mutations must be called only after vLLM has confirmed the corresponding
    load or unload, so this class stays in sync with actual GPU state.

    Args:
        max_loras: Maximum number of adapters that can be loaded simultaneously.
            Must match the ``--max-loras`` flag passed to vLLM.

    Raises:
        ValueError: If max_loras is less than 1.
    """

    def __init__(self, max_loras: int):
        if max_loras < 1:
            raise ValueError("max_loras must be >= 1")
        self.max_loras = max_loras
        self._loaded: dict[str, LoadedAdapter] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def loaded_names(self) -> list[str]:
        return list(self._loaded.keys())

    def get(self, name: str) -> LoadedAdapter | None:
        return self._loaded.get(name)

    def all_loaded(self) -> dict[str, LoadedAdapter]:
        return dict(self._loaded)

    def at_capacity(self) -> bool:
        return len(self._loaded) >= self.max_loras

    def record_load(self, name: str, meta: AdapterMeta) -> None:
        """Register that vLLM has successfully loaded an adapter.

        Args:
            name: Adapter name.
            meta: Registry metadata for the adapter.
        """
        now = datetime.now(UTC)
        self._loaded[name] = LoadedAdapter(name=name, meta=meta, loaded_at=now, last_used=now)

    def record_unload(self, name: str) -> None:
        """Register that vLLM has unloaded an adapter.

        Args:
            name: Adapter name to remove from the resident set.
        """
        self._loaded.pop(name, None)

    def touch(self, name: str) -> None:
        """Update last-used time and increment use count after an inference call.

        Args:
            name: Adapter name that was just used.
        """
        if name in self._loaded:
            adapter = self._loaded[name]
            self._loaded[name] = adapter.model_copy(
                update={"last_used": datetime.now(UTC), "use_count": adapter.use_count + 1}
            )

    def eviction_candidate(self) -> str | None:
        """Return the name of the least-recently-used loaded adapter.

        Returns:
            Adapter name, or None if nothing is currently loaded.
        """
        if not self._loaded:
            return None
        return min(self._loaded, key=lambda n: self._loaded[n].last_used)
