from __future__ import annotations

from typing import Any

import httpx

from .base import AdapterError, Generation, ModelAdapter


DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekError(AdapterError):
    pass


class DeepSeekAdapter(ModelAdapter):
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 90.0,
        app_title: str = "HumanBench",
        app_url: str | None = None,
    ) -> None:
        if not api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY is missing.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": app_title,
        }
        if app_url:
            headers["HTTP-Referer"] = app_url

        self._client = httpx.Client(timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DeepSeekAdapter":
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
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # Thinking mode is enabled by default on DeepSeek. We only specify it when
        # we need to control the effort explicitly.
        thinking = _thinking_mode(reasoning)
        if thinking is not None:
            body["thinking"] = thinking

        reasoning_effort = _reasoning_effort(reasoning)
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        if response_format is not None:
            body["response_format"] = response_format

        # DeepSeek ignores temperature in thinking mode, but keeping it out avoids
        # pretending we are doing something meaningful with a knob that won't bite.
        if thinking is not None and thinking.get("type") == "disabled":
            body["temperature"] = temperature

        try:
            response = self._client.post(DEEPSEEK_CHAT_COMPLETIONS_URL, json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _short_response_text(exc.response)
            raise DeepSeekError(
                f"DeepSeek rejected the request ({exc.response.status_code}). {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise DeepSeekError(f"DeepSeek network error: {exc}") from exc
        except ValueError as exc:
            raise DeepSeekError("DeepSeek returned a non-JSON response.") from exc

        choices = data.get("choices") or []
        if not choices:
            raise DeepSeekError(
                f"DeepSeek returned no choices. {_response_debug_summary(data, model)}"
            )

        choice = choices[0]
        if choice.get("error"):
            raise DeepSeekError(
                f"DeepSeek returned a choice error. {_choice_debug_summary(choice, data, model)}"
            )

        message = choice.get("message") or {}
        content = _normalize_content(message.get("content") if message else choice.get("text"))
        if not content:
            if allow_empty:
                return Generation(
                    content="",
                    model=str(data.get("model") or model),
                    raw=data,
                    usage=data.get("usage"),
                )
            raise DeepSeekError(
                f"DeepSeek returned an empty response. {_choice_debug_summary(choice, data, model)}"
            )

        return Generation(
            content=content.strip(),
            model=str(data.get("model") or model),
            raw=data,
            usage=data.get("usage"),
        )


def _thinking_mode(reasoning: dict[str, Any] | None) -> dict[str, str] | None:
    if reasoning is None:
        return None
    return {"type": "enabled"}


def _reasoning_effort(reasoning: dict[str, Any] | None) -> str | None:
    if not reasoning:
        return None

    effort = str(reasoning.get("effort", "")).strip().lower()
    if not effort or effort == "auto":
        return None
    if effort in {"high", "max"}:
        return effort
    if effort in {"xhigh"}:
        return "max"
    if effort in {"low", "medium"}:
        return "high"
    raise DeepSeekError(f"Invalid DeepSeek reasoning_effort: {effort!r}")


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return ""


def _short_response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    if len(text) > 700:
        text = text[:700] + "..."
    return text


def _choice_debug_summary(choice: dict[str, Any], data: dict[str, Any], requested_model: str) -> str:
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    choice_error = choice.get("error") or {}
    bits = [
        f"requested_model={requested_model}",
        f"returned_model={data.get('model')}",
        f"finish_reason={choice.get('finish_reason')}",
        f"native_finish_reason={choice.get('native_finish_reason')}",
        f"completion_tokens={usage.get('completion_tokens')}",
    ]
    if choice_error:
        bits.append(f"choice_error={choice_error}")
    if isinstance(message, dict):
        visible_keys = ", ".join(sorted(str(key) for key in message.keys())) or "none"
        bits.append(f"message_keys={visible_keys}")
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            bits.append(f"reasoning_chars={len(reasoning_content)}")
    return "(" + "; ".join(bits) + ")"


def _response_debug_summary(data: dict[str, Any], requested_model: str) -> str:
    usage = data.get("usage") or {}
    return (
        f"(requested_model={requested_model}; returned_model={data.get('model')}; "
        f"completion_tokens={usage.get('completion_tokens')})"
    )
