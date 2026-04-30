from __future__ import annotations

from typing import Any

import httpx

from .base import AdapterError, Generation, ModelAdapter


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(AdapterError):
    pass


class EmptyGenerationError(OpenRouterError):
    pass


class OpenRouterAdapter(ModelAdapter):
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 90.0,
        app_title: str = "HumanBench",
        app_url: str | None = None,
    ) -> None:
        if not api_key:
            raise OpenRouterError("OPENROUTER_API_KEY est manquante.")

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

    def __enter__(self) -> "OpenRouterAdapter":
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
            "temperature": temperature,
            # OpenRouter's strict provider routing currently matches endpoint support on "max_tokens".
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if reasoning is not None:
            body["reasoning"] = reasoning
        if provider is not None:
            body["provider"] = provider

        try:
            response = self._client.post(OPENROUTER_CHAT_COMPLETIONS_URL, json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _short_response_text(exc.response)
            raise OpenRouterError(
                f"OpenRouter a refuse la requete ({exc.response.status_code}). {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"Erreur reseau OpenRouter: {exc}") from exc
        except ValueError as exc:
            raise OpenRouterError("OpenRouter a renvoye une reponse non JSON.") from exc

        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError(
                f"OpenRouter n'a renvoye aucun choix. {_response_debug_summary(data, model)}"
            )

        choice = choices[0]
        if choice.get("error"):
            raise OpenRouterError(
                f"OpenRouter a renvoye une erreur de choix. {_choice_debug_summary(choice, data, model)}"
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
            raise EmptyGenerationError(
                f"OpenRouter a renvoye une reponse vide. {_choice_debug_summary(choice, data, model)}"
            )

        return Generation(
            content=content.strip(),
            model=str(data.get("model") or model),
            raw=data,
            usage=data.get("usage"),
        )


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
        f"modele_demande={requested_model}",
        f"modele_recu={data.get('model')}",
        f"finish_reason={choice.get('finish_reason')}",
        f"native_finish_reason={choice.get('native_finish_reason')}",
        f"completion_tokens={usage.get('completion_tokens')}",
    ]
    if choice_error:
        bits.append(f"choice_error={choice_error}")
    if isinstance(message, dict):
        visible_keys = ", ".join(sorted(str(key) for key in message.keys())) or "aucune"
        bits.append(f"message_keys={visible_keys}")
    return "(" + "; ".join(bits) + ")"


def _response_debug_summary(data: dict[str, Any], requested_model: str) -> str:
    usage = data.get("usage") or {}
    return (
        f"(modele_demande={requested_model}; modele_recu={data.get('model')}; "
        f"completion_tokens={usage.get('completion_tokens')})"
    )
