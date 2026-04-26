"""EntropyProbe runtime package."""

from __future__ import annotations

__all__ = ["GameLoop", "MemoryBus"]


def __getattr__(name: str):
    if name == "GameLoop":
        from .game_loop import GameLoop

        return GameLoop
    if name == "MemoryBus":
        from .memory_bus import MemoryBus

        return MemoryBus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
