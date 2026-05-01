from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .base import AdapterError, Generation, ModelAdapter


GOOGLE_GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GoogleError(AdapterError):
    pass


class GoogleAdapter(ModelAdapter):
    def __init__(self, api_key: str, *, timeout: float = 90.0) -> None:
        if not api_key:
            raise GoogleError("GOOGLE_API_KEY or GEMINI_API_KEY is missing.")

        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout, headers={"Content-Type": "application/json"})

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GoogleAdapter":
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
        system_instruction, contents = _convert_messages(messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if response_format and response_format.get("type") == "json_object":
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = GOOGLE_GENERATE_CONTENT_URL.format(model=quote(_strip_model_prefix(model), safe=""))
        try:
            response = self._client.post(url, params={"key": self._api_key}, json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _short_response_text(exc.response)
            raise GoogleError(f"Google Gemini rejected the request ({exc.response.status_code}). {detail}") from exc
        except httpx.HTTPError as exc:
            raise GoogleError(f"Google Gemini network error: {exc}") from exc
        except ValueError as exc:
            raise GoogleError("Google Gemini returned a non-JSON response.") from exc

        content = _extract_text(data)
        if not content:
            if allow_empty:
                return Generation("", model, data, data.get("usageMetadata"))
            raise GoogleError(f"Google Gemini returned an empty response. {_response_debug_summary(data, model)}")

        return Generation(content.strip(), model, data, data.get("usageMetadata"))


def _convert_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        text = message.get("content", "")
        if role == "system":
            system_parts.append(text)
            continue
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            }
        )

    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    return ("\n\n".join(part for part in system_parts if part).strip() or None), contents


def _extract_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = (candidates[0].get("content") or {}).get("parts") or []
    parts: list[str] = []
    for part in content:
        text = part.get("text") if isinstance(part, dict) else None
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _strip_model_prefix(model: str) -> str:
    return model.removeprefix("models/")


def _short_response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:700] + "..." if len(text) > 700 else text


def _response_debug_summary(data: dict[str, Any], requested_model: str) -> str:
    usage = data.get("usageMetadata") or {}
    candidates = data.get("candidates") or []
    finish_reason = candidates[0].get("finishReason") if candidates else None
    return (
        f"(requested_model={requested_model}; finish_reason={finish_reason}; "
        f"candidates={len(candidates)}; output_tokens={usage.get('candidatesTokenCount')})"
    )
