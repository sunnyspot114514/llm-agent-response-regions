"""EntropyProbe agents package."""

from __future__ import annotations

__all__ = ["BaseAgent", "LLMAgent", "LazyAgent"]


def __getattr__(name: str):
    if name == "BaseAgent":
        from .base_agent import BaseAgent

        return BaseAgent
    if name == "LLMAgent":
        from .llm_agent import LLMAgent

        return LLMAgent
    if name == "LazyAgent":
        from .lazy_agent import LazyAgent

        return LazyAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
