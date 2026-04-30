from __future__ import annotations

from typing import Any

import httpx

from .base import AdapterError, Generation, ModelAdapter


MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralError(AdapterError):
    pass


class MistralAdapter(ModelAdapter):
    def __init__(self, api_key: str, *, timeout: float = 90.0) -> None:
        if not api_key:
            raise MistralError("MISTRAL_API_KEY est manquante.")

        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MistralAdapter":
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
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        try:
            response = self._client.post(MISTRAL_CHAT_COMPLETIONS_URL, json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = _short_response_text(exc.response)
            raise MistralError(f"Mistral a refuse la requete ({exc.response.status_code}). {detail}") from exc
        except httpx.HTTPError as exc:
            raise MistralError(f"Erreur reseau Mistral: {exc}") from exc
        except ValueError as exc:
            raise MistralError("Mistral a renvoye une reponse non JSON.") from exc

        choices = data.get("choices") or []
        if not choices:
            raise MistralError(f"Mistral n'a renvoye aucun choix. {_response_debug_summary(data, model)}")

        choice = choices[0]
        message = choice.get("message") or {}
        content = _normalize_content(message.get("content") if message else choice.get("text"))
        if not content:
            if allow_empty:
                return Generation("", str(data.get("model") or model), data, data.get("usage"))
            raise MistralError(f"Mistral a renvoye une reponse vide. {_choice_debug_summary(choice, data, model)}")

        return Generation(content.strip(), str(data.get("model") or model), data, data.get("usage"))


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
    return text[:700] + "..." if len(text) > 700 else text


def _choice_debug_summary(choice: dict[str, Any], data: dict[str, Any], requested_model: str) -> str:
    usage = data.get("usage") or {}
    return (
        f"(modele_demande={requested_model}; modele_recu={data.get('model')}; "
        f"finish_reason={choice.get('finish_reason')}; completion_tokens={usage.get('completion_tokens')})"
    )


def _response_debug_summary(data: dict[str, Any], requested_model: str) -> str:
    usage = data.get("usage") or {}
    return (
        f"(modele_demande={requested_model}; modele_recu={data.get('model')}; "
        f"completion_tokens={usage.get('completion_tokens')})"
    )
