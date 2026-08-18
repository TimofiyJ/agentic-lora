"""Adapter registry backed by a YAML config file."""

import yaml
from pathlib import Path
from .models import AdapterMeta


class AdapterRegistry:
    """YAML-backed registry mapping adapter names to their metadata.

    The expected config structure is::

        adapters:
          sql_adapter:
            path: /path/to/adapter
            description: "Generates SQL from natural language"
            rank: 16
            base_model: meta-llama/Llama-3.2-3B-Instruct
            tags: [sql, code]

    Args:
        config_path: Path to the adapters YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist at startup.
    """

    def __init__(self, config_path: str | Path):
        self._path = Path(config_path)
        self._adapters: dict[str, AdapterMeta] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Adapter config not found: {self._path}")
        with open(self._path) as f:
            data = yaml.safe_load(f)
        self._adapters = {
            name: AdapterMeta(**meta)
            for name, meta in data.get("adapters", {}).items()
        }

    def reload(self) -> None:
        """Re-read the YAML file without restarting the service."""
        self._adapters = {}
        self._load()

    def get(self, name: str) -> AdapterMeta | None:
        """Return metadata for an adapter, or None if not registered.

        Args:
            name: Adapter name as defined in the YAML config.

        Returns:
            AdapterMeta if found, None otherwise.
        """
        return self._adapters.get(name)

    def all(self) -> dict[str, AdapterMeta]:
        """Return a snapshot of all registered adapters."""
        return dict(self._adapters)

    def __contains__(self, name: str) -> bool:
        return name in self._adapters
