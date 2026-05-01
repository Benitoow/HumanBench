from __future__ import annotations

from typing import Any

import httpx

from .base import AdapterError, Generation, ModelAdapter


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicError(AdapterError):
    pass


class AnthropicAdapter(ModelAdapter):
    def __init__(self, api_key: str, *, timeout: float = 90.0) -> None:
        if not api_key:
            raise AnthropicError("ANTHROPIC_API_KEY is missing.")

        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AnthropicAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
        system, anthropic_messages = _convert_messages(messages)
        body: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            body["system"] = system

        try:
            response = self._client.post(ANTHROPIC_MESSAGES_URL, json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _short_response_text(exc.response)
            raise AnthropicError(f"Anthropic rejected the request ({exc.response.status_code}). {detail}") from exc
        except httpx.HTTPError as exc:
            raise AnthropicError(f"Anthropic network error: {exc}") from exc
        except ValueError as exc:
            raise AnthropicError("Anthropic returned a non-JSON response.") from exc

        content = _normalize_content(data.get("content"))
        if not content:
            if allow_empty:
                return Generation("", str(data.get("model") or model), data, data.get("usage"))
            raise AnthropicError(f"Anthropic returned an empty response. {_response_debug_summary(data, model)}")

        return Generation(content.strip(), str(data.get("model") or model), data, data.get("usage"))


def _convert_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    system_parts: list[str] = []
    converted: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append({"role": "user", "content": content})

    if not converted:
        converted.append({"role": "user", "content": ""})

    return ("\n\n".join(part for part in system_parts if part).strip() or None), converted


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _short_response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:700] + "..." if len(text) > 700 else text


def _response_debug_summary(data: dict[str, Any], requested_model: str) -> str:
    usage = data.get("usage") or {}
    return (
        f"(requested_model={requested_model}; returned_model={data.get('model')}; "
        f"stop_reason={data.get('stop_reason')}; output_tokens={usage.get('output_tokens')})"
    )
