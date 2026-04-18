from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters import BackendAdapter, CVXPYAdapter, NumPyAdapter, SciPySparseAdapter


@dataclass
class BackendRegistry:
    _adapters: list[BackendAdapter] = field(default_factory=list)

    def register(self, adapter: BackendAdapter) -> None:
        self._adapters.append(adapter)
        self._adapters.sort(key=lambda item: item.priority, reverse=True)

    @property
    def adapters(self) -> tuple[BackendAdapter, ...]:
        return tuple(self._adapters)

    def find_adapter(self, obj: Any) -> BackendAdapter:
        for adapter in self._adapters:
            if adapter.can_handle(obj):
                return adapter
        raise TypeError(f"no registered backend can handle objects of type {type(obj)!r}")


backend_registry = BackendRegistry()
backend_registry.register(NumPyAdapter())
backend_registry.register(SciPySparseAdapter())
backend_registry.register(CVXPYAdapter())


def resolve_backend(*objs: Any) -> BackendAdapter:
    selected: BackendAdapter | None = None
    for obj in objs:
        adapter = backend_registry.find_adapter(obj)
        if selected is None or adapter.priority > selected.priority:
            selected = adapter
    if selected is None:
        raise TypeError("at least one object is required to resolve a backend")
    return selected
