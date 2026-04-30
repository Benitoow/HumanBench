from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class AdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class Generation:
    content: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None


class ModelAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> Generation:
        raise NotImplementedError
