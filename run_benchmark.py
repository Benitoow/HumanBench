from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from adapters import (
    AdapterError,
    AnthropicAdapter,
    DeepSeekAdapter,
    Generation,
    GoogleAdapter,
    MistralAdapter,
    ModelAdapter,
    OpenAIAdapter,
    OpenRouterAdapter,
)


VERSION = "0.1.0"
DEFAULT_PROMPTS_PATH = Path("prompts.json")
DEFAULT_JUDGE_PROMPT_PATH = Path("judge/prompt.txt")
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_PROVIDER = "auto"
TESTED_MODEL_MAX_TOKENS = 900
DEFAULT_JUDGE_MAX_TOKENS = 8192
MODEL_GENERATION_ATTEMPTS = 3
JUDGE_GENERATION_ATTEMPTS = 3
DEEPSEEK_V4_PRO_MODEL = "deepseek/deepseek-v4-pro"
DEEPSEEK_OFFICIAL_PROVIDER = "deepseek"
OPENROUTER_BACKEND = "openrouter"
DEEPSEEK_BACKEND = "deepseek"
ANTHROPIC_BACKEND = "anthropic"
OPENAI_BACKEND = "openai"
MISTRAL_BACKEND = "mistral"
GOOGLE_BACKEND = "google"
SUPPORTED_BACKENDS = {
    DEFAULT_PROVIDER,
    OPENROUTER_BACKEND,
    DEEPSEEK_BACKEND,
    ANTHROPIC_BACKEND,
    OPENAI_BACKEND,
    MISTRAL_BACKEND,
    GOOGLE_BACKEND,
}

# (provider_id, display_name, key_env_name, preset_models)
_WIZARD_PROVIDERS: list[tuple[str, str, str | None, list[str]]] = [
    (
        ANTHROPIC_BACKEND, "Anthropic", "ANTHROPIC_API_KEY",
        ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
    ),
    (
        OPENAI_BACKEND, "OpenAI", "OPENAI_API_KEY",
        ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3", "o4-mini"],
    ),
    (
        MISTRAL_BACKEND, "Mistral", "MISTRAL_API_KEY",
        ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "magistral-latest"],
    ),
    (
        GOOGLE_BACKEND, "Google Gemini", "GOOGLE_API_KEY",
        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    ),
    (
        DEEPSEEK_BACKEND, "DeepSeek", "DEEPSEEK_API_KEY",
        ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"],
    ),
    (
        OPENROUTER_BACKEND, "OpenRouter", "OPENROUTER_API_KEY",
        [
            "anthropic/claude-sonnet-4-6",
            "openai/gpt-4.1",
            "google/gemini-2.5-flash",
            "mistralai/mistral-large",
            "deepseek/deepseek-v4-pro",
        ],
    ),
    ("custom", "Autre / Custom", None, []),
]


class BenchmarkError(RuntimeError):
    pass


class BlockBarColumn(ProgressColumn):
    def __init__(
        self,
        *,
        width: int = 32,
        complete_style: str = "green",
        remaining_style: str = "grey23",
    ) -> None:
        super().__init__()
        self.width = width
        self.complete_style = complete_style
        self.remaining_style = remaining_style

    def render(self, task: Any) -> Text:
        total = float(task.total or 0)
        completed = float(task.completed or 0)
        ratio = 0.0 if total <= 0 else min(1.0, max(0.0, completed / total))
        filled = int(round(self.width * ratio))
        return render_block_bar(
            filled=filled,
            width=self.width,
            complete_style=self.complete_style,
            remaining_style=self.remaining_style,
        )


@dataclass(frozen=True)
class PromptItem:
    id: str
    type: str
    prompt: str
    reference_humaine: str | None = None
    reference_score: int | None = None
    notes_juge: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any], index: int) -> "PromptItem":
        missing = [key for key in ("id", "type", "prompt") if not raw.get(key)]
        if missing:
            raise BenchmarkError(
                f"Prompt #{index + 1}: missing required field(s): {', '.join(missing)}"
            )
        return cls(
            id=str(raw["id"]),
            type=str(raw["type"]),
            prompt=str(raw["prompt"]),
            reference_humaine=_optional_string(raw.get("reference_humaine")),
            reference_score=_optional_int(raw.get("reference_score")),
            notes_juge=_optional_string(raw.get("notes_juge")),
        )


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    max_tokens: int
    reasoning: dict[str, Any] | None
    provider: dict[str, Any] | None

    @property
    def reasoning_label(self) -> str:
        if not self.reasoning:
            return "auto"
        effort = self.reasoning.get("effort", "auto")
        exclude = self.reasoning.get("exclude")
        visibility = "reasoning hidden" if exclude else "reasoning kept"
        return f"{effort} ({visibility})"

    @property
    def provider_label(self) -> str:
        if not self.provider:
            return "OpenRouter auto"
        order = self.provider.get("order") or []
        fallbacks = self.provider.get("allow_fallbacks")
        provider_text = ", ".join(str(item) for item in order) if order else "auto order"
        fallback_text = "fallbacks on" if fallbacks else "fallbacks off"
        return f"{provider_text} ({fallback_text})"

    def to_report(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "provider": self.provider,
        }


@dataclass(frozen=True)
class BackendConfig:
    name: str
    api_key: str
    api_key_source: str

    @property
    def label(self) -> str:
        return {
            OPENROUTER_BACKEND: "OpenRouter",
            DEEPSEEK_BACKEND: "DeepSeek",
            ANTHROPIC_BACKEND: "Anthropic",
            OPENAI_BACKEND: "OpenAI",
            MISTRAL_BACKEND: "Mistral",
            GOOGLE_BACKEND: "Google Gemini",
        }.get(self.name, self.name)


@dataclass(frozen=True)
class Score:
    format: int
    densite: int
    ton: int
    total: int
    commentaire_format: str
    commentaire_densite: str
    commentaire_ton: str
    verdict: str

    @classmethod
    def from_judge_json(cls, payload: dict[str, Any]) -> "Score":
        format_score = _clamp_int(payload.get("format"), 0, 33, "format")
        densite_score = _clamp_int(
            payload.get("densite", payload.get("densité")), 0, 33, "densite"
        )
        ton_score = _clamp_int(payload.get("ton"), 0, 34, "ton")
        total = format_score + densite_score + ton_score

        return cls(
            format=format_score,
            densite=densite_score,
            ton=ton_score,
            total=total,
            commentaire_format=str(payload.get("commentaire_format", "")).strip(),
            commentaire_densite=str(payload.get("commentaire_densite", "")).strip(),
            commentaire_ton=str(payload.get("commentaire_ton", "")).strip(),
            verdict=str(payload.get("verdict", "")).strip(),
        )

    @classmethod
    def empty_model_response(cls) -> "Score":
        return cls(
            format=0,
            densite=0,
            ton=0,
            total=0,
            commentaire_format="No response to evaluate.",
            commentaire_densite="No measurable density.",
            commentaire_ton="No observable tone.",
            verdict="The tested model returned no text.",
        )


class RichArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, console: Console, **kwargs: Any) -> None:
        self.console = console
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        self.console.print(
            Panel(
                Text(message, style="bold red"),
                title="[bold red]Invalid argument[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        print_help(self.console)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    configure_terminal_encoding()
    console = Console()
    argv = sys.argv[1:] if argv is None else argv

    if any(arg in ("-h", "--help") for arg in argv):
        print_help(console)
        return 0

    parser = build_parser(console)
    args = parser.parse_args(argv)

    try:
        run(args, console)
    except KeyboardInterrupt:
        console.print(
            Panel(
                "Benchmark interrupted. Fine. Humans are allowed to pull the plug.",
                title="[bold yellow]Stop[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        return 130
    except (BenchmarkError, AdapterError) as exc:
        console.print(
            Panel(
                str(exc),
                title="[bold red]Benchmark blocked[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return 1

    return 0


def configure_terminal_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


configure_terminal_encoding()


def build_parser(console: Console) -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog="python run_benchmark.py",
        add_help=False,
        console=console,
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to test. Defaults to MODEL_TESTED from .env.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=sorted(SUPPORTED_BACKENDS),
        help="API backend to use. Defaults to auto.",
    )
    parser.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS_PATH),
        help="Path to the prompt JSON file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full model responses while the benchmark runs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the JSON report.",
    )
    return parser


def print_help(console: Console) -> None:
    usage = Table.grid(padding=(0, 1))
    usage.add_column(style="bold cyan", no_wrap=True)
    usage.add_column(style="white")
    usage.add_row("Command", "python run_benchmark.py --model anthropic/claude-sonnet-4-6")
    usage.add_row("Purpose", "Test one model through an API backend and score its humanness.")

    options = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
    )
    options.add_column("Option", style="cyan", no_wrap=True)
    options.add_column("Description", style="white")
    options.add_row("--model", "Model to test. Falls back to MODEL_TESTED in .env.")
    options.add_row("--provider", "Backend for the tested model: auto, openrouter, anthropic, openai, mistral, google, or deepseek.")
    options.add_row("--prompts", "Custom prompt JSON file. Defaults to prompts.json.")
    options.add_row("--verbose", "Print each full model response in real time.")
    options.add_row("--output", "JSON report path. Defaults to results/<model>_<date>.json.")
    options.add_row("-h, --help", "Show this Rich help screen. The gray 1998 version has suffered enough.")

    console.print(render_banner())
    console.print(
        Panel(
            Group(usage, Text(""), options),
            title="[bold cyan]Phase 1 - terminal MVP[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def run(args: argparse.Namespace, console: Console) -> None:
    load_dotenv()

    requested_model = _optional_string(args.model) or _optional_string(os.getenv("MODEL_TESTED"))
    if not requested_model:
        raise BenchmarkError("No tested model configured. Use --model or MODEL_TESTED in .env.")
    judge_model = DEFAULT_JUDGE_MODEL
    model_backend_config = build_backend_config(
        requested_provider=_optional_string(args.provider) or _optional_string(os.getenv("PROVIDER")),
        model=requested_model,
        role="tested model",
    )
    judge_backend_config = build_backend_config(
        requested_provider=None,
        model=judge_model,
        role="judge",
    )
    backend_model = normalize_model_for_backend(requested_model, model_backend_config.name)
    backend_judge_model = normalize_model_for_backend(judge_model, judge_backend_config.name)
    judge_config = build_judge_config(backend_judge_model, judge_backend_config.name)
    prompts_path = Path(args.prompts)
    judge_prompt_path = DEFAULT_JUDGE_PROMPT_PATH
    output_path = Path(args.output) if args.output else default_output_path(backend_model)

    prompts = load_prompts(prompts_path)
    judge_prompt = load_text(judge_prompt_path, "judge prompt")

    console.print(render_banner())
    console.print(
        render_config(
            model_backend_config,
            backend_model,
            judge_backend_config,
            judge_config,
            prompts_path,
            output_path,
            len(prompts),
        )
    )

    results: list[dict[str, Any]] = []

    progress = Progress(
        TextColumn("[bold cyan]HB[/bold cyan]"),
        TextColumn("[bold cyan]{task.description}"),
        BlockBarColumn(width=24, complete_style="green"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with ExitStack() as stack:
        model_adapter = stack.enter_context(build_adapter(model_backend_config))
        judge_adapter = stack.enter_context(build_adapter(judge_backend_config))
        with progress:
            task_id = progress.add_task("Initialisation", total=len(prompts))

            for item in prompts:
                progress.update(task_id, description=f"Prompt {item.id}")
                model_completion = generate_answer(model_adapter, backend_model, item)
                response_text = model_completion.content

                if args.verbose:
                    console.print(render_verbose_answer(item, response_text))

                if not response_text.strip():
                    score = Score.empty_model_response()
                    result = build_prompt_result(
                        item,
                        response_text,
                        score,
                        raw_judge_text="",
                        raw_judge_json={},
                        model_raw_json=model_completion.raw,
                        error={
                            "stage": "model",
                            "type": "empty_response",
                            "message": "The tested model returned no text after multiple attempts.",
                        },
                    )
                    results.append(result)
                    console.print(render_result_row(item, score))
                    progress.advance(task_id)
                    continue

                score, raw_judge_text, raw_judge_json = judge_answer(
                    judge_adapter,
                    judge_config,
                    judge_prompt,
                    item,
                    response_text,
                )

                result = build_prompt_result(
                    item,
                    response_text,
                    score,
                    raw_judge_text,
                    raw_judge_json,
                    model_raw_json=model_completion.raw,
                )
                results.append(result)
                console.print(render_result_row(item, score))
                progress.advance(task_id)

    report = build_report(
        model_backend_config,
        judge_backend_config,
        requested_model,
        backend_model,
        backend_judge_model,
        judge_config,
        prompts_path,
        output_path,
        results,
    )
    write_report(output_path, report)
    sync_leaderboard(console, report)

    console.print(render_final_summary(report, output_path))


def build_backend_config(
    *,
    requested_provider: str | None,
    model: str,
    role: str,
    fallback_provider: str | None = None,
) -> BackendConfig:
    requested_provider = (requested_provider or DEFAULT_PROVIDER).strip().lower()
    if requested_provider not in SUPPORTED_BACKENDS:
        raise BenchmarkError(
            f"Invalid PROVIDER for {role}: {requested_provider!r}. Use auto, openrouter, anthropic, openai, mistral, google, or deepseek."
        )

    if requested_provider == DEFAULT_PROVIDER:
        inferred = infer_backend_for_model(model)
        if inferred is not None:
            requested_provider = inferred
        elif fallback_provider is not None:
            requested_provider = fallback_provider
        else:
            requested_provider = _infer_backend_from_keys()
            if requested_provider is None:
                raise BenchmarkError(
                    f"Could not infer a backend for {role}. Set PROVIDER or add a provider API key."
                )

    api_key, source = resolve_api_key(requested_provider)
    if not api_key:
        env_name = preferred_key_name(requested_provider)
        raise BenchmarkError(f"{env_name} is missing for {role}. Add it to .env or use API_KEY with an explicit PROVIDER.")
    return BackendConfig(name=requested_provider, api_key=api_key, api_key_source=source)

    raise BenchmarkError(f"Unknown backend: {requested_provider!r}")


def build_adapter(backend_config: BackendConfig) -> ModelAdapter:
    if backend_config.name == OPENROUTER_BACKEND:
        return OpenRouterAdapter(backend_config.api_key)
    if backend_config.name == DEEPSEEK_BACKEND:
        return DeepSeekAdapter(backend_config.api_key)
    if backend_config.name == ANTHROPIC_BACKEND:
        return AnthropicAdapter(backend_config.api_key)
    if backend_config.name == OPENAI_BACKEND:
        return OpenAIAdapter(backend_config.api_key)
    if backend_config.name == MISTRAL_BACKEND:
        return MistralAdapter(backend_config.api_key)
    if backend_config.name == GOOGLE_BACKEND:
        return GoogleAdapter(backend_config.api_key)
    raise BenchmarkError(f"Unsupported backend: {backend_config.name!r}")


def infer_backend_for_model(model: str) -> str | None:
    model = model.strip().lower()
    if not model:
        return None
    if model.startswith("deepseek/"):
        return OPENROUTER_BACKEND
    if model.startswith(("anthropic/", "openai/", "mistralai/", "google/")):
        return OPENROUTER_BACKEND
    if model in {
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-chat",
        "deepseek-reasoner",
    }:
        return DEEPSEEK_BACKEND
    if model.startswith("claude-"):
        return ANTHROPIC_BACKEND
    if model.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt-")):
        return OPENAI_BACKEND
    if model.startswith(("mistral-", "ministral-", "codestral-", "magistral-")):
        return MISTRAL_BACKEND
    if model.startswith(("gemini-", "models/gemini-")):
        return GOOGLE_BACKEND
    if "/" in model:
        return OPENROUTER_BACKEND
    return None


def _infer_backend_from_keys() -> str | None:
    for provider in (
        OPENROUTER_BACKEND,
        DEEPSEEK_BACKEND,
        ANTHROPIC_BACKEND,
        OPENAI_BACKEND,
        MISTRAL_BACKEND,
        GOOGLE_BACKEND,
    ):
        api_key, _ = resolve_api_key(provider, allow_generic=False)
        if api_key:
            return provider
    return None


def resolve_api_key(provider: str, *, allow_generic: bool = True) -> tuple[str | None, str]:
    env_names = {
        OPENROUTER_BACKEND: ("OPENROUTER_API_KEY",),
        DEEPSEEK_BACKEND: ("DEEPSEEK_API_KEY",),
        ANTHROPIC_BACKEND: ("ANTHROPIC_API_KEY",),
        OPENAI_BACKEND: ("OPENAI_API_KEY",),
        MISTRAL_BACKEND: ("MISTRAL_API_KEY",),
        GOOGLE_BACKEND: ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    }.get(provider, ())

    for env_name in env_names:
        value = _optional_string(os.getenv(env_name))
        if value:
            return value, env_name

    if allow_generic:
        value = _optional_string(os.getenv("API_KEY"))
        if value:
            return value, "API_KEY"

    return None, preferred_key_name(provider)


def preferred_key_name(provider: str) -> str:
    return {
        OPENROUTER_BACKEND: "OPENROUTER_API_KEY",
        DEEPSEEK_BACKEND: "DEEPSEEK_API_KEY",
        ANTHROPIC_BACKEND: "ANTHROPIC_API_KEY",
        OPENAI_BACKEND: "OPENAI_API_KEY",
        MISTRAL_BACKEND: "MISTRAL_API_KEY",
        GOOGLE_BACKEND: "GOOGLE_API_KEY",
    }.get(provider, "API_KEY")


def normalize_model_for_backend(model: str, backend_name: str) -> str:
    model = model.strip()
    if backend_name == OPENROUTER_BACKEND:
        if model in {
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "deepseek-chat",
            "deepseek-reasoner",
        }:
            return f"deepseek/{model}"
        return model

    if backend_name == DEEPSEEK_BACKEND:
        if model.startswith("deepseek/"):
            return model.split("/", 1)[1]
        if "/" in model:
            raise BenchmarkError(
                "The DeepSeek backend expects native model names like `deepseek-v4-pro`, not OpenRouter slugs."
            )
        return model

    if backend_name == ANTHROPIC_BACKEND:
        if model.startswith("anthropic/"):
            return model.split("/", 1)[1]
        if "/" in model:
            raise BenchmarkError("The Anthropic backend expects a native model name like `claude-sonnet-4-5`, not an OpenRouter slug.")
        return model

    if backend_name == OPENAI_BACKEND:
        if model.startswith("openai/"):
            return model.split("/", 1)[1]
        if "/" in model:
            raise BenchmarkError("The OpenAI backend expects a native model name like `gpt-5.2`, not an OpenRouter slug.")
        return model

    if backend_name == MISTRAL_BACKEND:
        if model.startswith(("mistralai/", "mistral/")):
            return model.split("/", 1)[1]
        if "/" in model:
            raise BenchmarkError("The Mistral backend expects a native model name like `mistral-large-latest`, not an OpenRouter slug.")
        return model

    if backend_name == GOOGLE_BACKEND:
        if model.startswith("google/"):
            return model.split("/", 1)[1]
        if "/" in model and not model.startswith("models/"):
            raise BenchmarkError("The Google backend expects a native model name like `gemini-2.5-flash`, not an OpenRouter slug.")
        return model

    return model


def generate_answer(adapter: ModelAdapter, model: str, item: PromptItem) -> Generation:
    last_completion: Generation | None = None
    last_error: AdapterError | None = None

    for attempt in range(1, MODEL_GENERATION_ATTEMPTS + 1):
        try:
            completion = adapter.generate(
                [{"role": "user", "content": item.prompt}],
                model=model,
                temperature=0.7,
                max_tokens=TESTED_MODEL_MAX_TOKENS,
                allow_empty=True,
            )
        except AdapterError as exc:
            last_error = exc
            if attempt < MODEL_GENERATION_ATTEMPTS:
                time.sleep(0.8 * attempt)
                continue
            raise AdapterError(f"{item.id} / tested model {model}: {exc}") from exc

        last_completion = completion
        if completion.content.strip():
            return completion
        if attempt < MODEL_GENERATION_ATTEMPTS:
            time.sleep(0.8 * attempt)

    if last_completion is not None:
        return last_completion

    raise AdapterError(f"{item.id} / tested model {model}: {last_error}")


def build_judge_config(judge_model: str, backend_name: str) -> JudgeConfig:
    normalized_judge = judge_model.lower()
    default_deepseek_v4 = normalized_judge in {DEEPSEEK_V4_PRO_MODEL, "deepseek-v4-pro"}

    max_tokens = DEFAULT_JUDGE_MAX_TOKENS
    if max_tokens < 256:
        raise BenchmarkError("DEFAULT_JUDGE_MAX_TOKENS must be >= 256. A judge with no budget is just vibes in a robe.")

    effort = "xhigh" if default_deepseek_v4 else None

    reasoning: dict[str, Any] | None = None
    if effort and effort.lower() != "auto":
        reasoning = {"effort": effort, "exclude": True}

    provider: dict[str, Any] | None = None
    if backend_name == OPENROUTER_BACKEND:
        provider_order = [DEEPSEEK_OFFICIAL_PROVIDER] if default_deepseek_v4 else []
        if provider_order:
            provider = {
                "order": provider_order,
                "allow_fallbacks": False,
                "require_parameters": True,
            }

    return JudgeConfig(
        model=judge_model,
        max_tokens=max_tokens,
        reasoning=reasoning,
        provider=provider,
    )


def judge_answer(
    adapter: ModelAdapter,
    judge_config: JudgeConfig,
    judge_prompt: str,
    item: PromptItem,
    response_text: str,
) -> tuple[Score, str, dict[str, Any]]:
    messages = build_judge_messages(judge_prompt, item, response_text)
    last_raw_text = ""
    last_error: Exception | None = None

    for attempt in range(JUDGE_GENERATION_ATTEMPTS):
        completion = adapter.generate(
            messages,
            model=judge_config.model,
            temperature=0,
            max_tokens=judge_config.max_tokens,
            response_format={"type": "json_object"},
            reasoning=judge_config.reasoning,
            provider=judge_config.provider,
            allow_empty=True,
        )
        last_raw_text = completion.content
        if not last_raw_text.strip():
            last_error = BenchmarkError("The judge returned no text.")
            if attempt < JUDGE_GENERATION_ATTEMPTS - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was empty. Return only the requested JSON object."
                        ),
                    }
                )
                time.sleep(0.8 * (attempt + 1))
                continue
            break

        try:
            raw_json = extract_json_object(last_raw_text)
            return Score.from_judge_json(raw_json), last_raw_text, raw_json
        except BenchmarkError as exc:
            last_error = exc
            messages.extend(
                [
                    {"role": "assistant", "content": last_raw_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was not a valid JSON object. "
                            "Return only the requested JSON object, with no Markdown and no extra commentary."
                        ),
                    },
                ]
            )

    raise BenchmarkError(f"The judge failed to return valid JSON. Last error: {last_error}")


def build_judge_messages(
    judge_prompt: str,
    item: PromptItem,
    response_text: str,
) -> list[dict[str, str]]:
    payload = {
        "id": item.id,
        "type": item.type,
        "user_prompt": item.prompt,
        "reference_humaine": item.reference_humaine,
        "reference_score": item.reference_score,
        "notes_juge": item.notes_juge,
        "ai_response_to_evaluate": response_text,
    }
    return [
        {"role": "system", "content": judge_prompt},
        {
            "role": "user",
            "content": (
                "Evaluate the AI response below using the HumanBench framework.\n"
                "The context JSON follows:\n\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def load_prompts(path: Path) -> list[PromptItem]:
    if not path.exists():
        raise BenchmarkError(f"Prompt file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid prompt file: {exc}") from exc

    if not isinstance(raw, list):
        raise BenchmarkError("The prompt file must contain a JSON array.")

    prompts = [PromptItem.from_raw(item, index) for index, item in enumerate(raw)]
    if not prompts:
        raise BenchmarkError("The prompt file is empty. Even a benchmark needs ammunition.")
    return prompts


def load_text(path: Path, label: str) -> str:
    if not path.exists():
        raise BenchmarkError(f"{label.capitalize()} not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise BenchmarkError(f"{label.capitalize()} is empty: {path}")
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    if fenced != text:
        candidates.append(fenced.strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise BenchmarkError("Could not extract a valid JSON object from the judge response.")


def build_prompt_result(
    item: PromptItem,
    response_text: str,
    score: Score,
    raw_judge_text: str,
    raw_judge_json: dict[str, Any],
    *,
    model_raw_json: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "id": item.id,
        "type": item.type,
        "prompt": item.prompt,
        "response": response_text,
        "score": {
            "format": score.format,
            "densite": score.densite,
            "ton": score.ton,
            "total": score.total,
        },
        "comments": {
            "format": score.commentaire_format,
            "densite": score.commentaire_densite,
            "ton": score.commentaire_ton,
            "verdict": score.verdict,
        },
        "reference": {
            "humaine": item.reference_humaine,
            "score": item.reference_score,
            "notes_juge": item.notes_juge,
        },
        "model_raw_json": model_raw_json,
        "judge_raw_text": raw_judge_text,
        "judge_raw_json": raw_judge_json,
    }
    if error is not None:
        result["error"] = error
    return result


def build_report(
    model_backend_config: BackendConfig,
    judge_backend_config: BackendConfig,
    requested_model: str,
    model: str,
    judge_model: str,
    judge_config: JudgeConfig,
    prompts_path: Path,
    output_path: Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize_scores(results)
    return {
        "humanbench_version": VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": model_backend_config.name,
        "judge_backend": judge_backend_config.name,
        "api_key_source": model_backend_config.api_key_source,
        "judge_api_key_source": judge_backend_config.api_key_source,
        "requested_model": requested_model,
        "model": model,
        "judge": judge_model,
        "judge_config": judge_config.to_report(),
        "prompts_path": str(prompts_path),
        "output_path": str(output_path),
        "summary": summary,
        "results": results,
    }


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "score_final": 0,
            "format": 0,
            "densite": 0,
            "ton": 0,
            "interpretation": "No prompts",
        }

    totals = [int(result["score"]["total"]) for result in results]
    formats = [int(result["score"]["format"]) for result in results]
    densites = [int(result["score"]["densite"]) for result in results]
    tons = [int(result["score"]["ton"]) for result in results]
    final_score = round(sum(totals) / len(totals))

    return {
        "score_final": final_score,
        "format": round(sum(formats) / len(formats)),
        "densite": round(sum(densites) / len(densites)),
        "ton": round(sum(tons) / len(tons)),
        "interpretation": score_label(final_score),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_leaderboard(console: Console, report: dict[str, Any]) -> None:
    """Synchronize benchmark results to leaderboard files."""
    results_path = Path("results.json")
    site_results_path = Path("site/results.json")
    
    # Create leaderboard entry from report
    entry = {
        "requested_model": report["requested_model"],
        "created_at": report["created_at"],
        "summary": report["summary"],
    }
    
    # Load existing results
    if results_path.exists():
        leaderboard = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        leaderboard = []
    
    # Add new entry
    leaderboard.append(entry)
    
    # Write to root results.json
    results_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Sync to site/results.json
    site_results_path.parent.mkdir(parents=True, exist_ok=True)
    site_results_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Check for WEB_REPO_PATH
    web_repo_path = os.getenv("WEB_REPO_PATH")
    if not web_repo_path:
        console.print(
            Panel(
                "[yellow]WARNING: WEB_REPO_PATH not configured in .env[/yellow]\n"
                "Results synchronized locally, but leaderboard repository not updated.\n"
                "Add [bold]WEB_REPO_PATH=/path/to/leaderboard/repo[/bold] to .env to auto-sync.",
                title="[bold yellow]Leaderboard Sync[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
    else:
        console.print(
            Panel(
                f"Results synchronized to {site_results_path}",
                title="[bold green]Leaderboard Synced[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )


def default_output_path(model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("results") / f"{safe_model}_{stamp}.json"


def render_banner() -> Panel:
    title = Text()
    title.append("██╗  ██╗██╗   ██╗███╗   ███╗ █████╗ ███╗   ██╗\n", style="bold cyan")
    title.append("██║  ██║██║   ██║████╗ ████║██╔══██╗████╗  ██║\n", style="bold cyan")
    title.append("███████║██║   ██║██╔████╔██║███████║██╔██╗ ██║\n", style="bold magenta")
    title.append("██╔══██║██║   ██║██║╚██╔╝██║██╔══██║██║╚██╗██║\n", style="bold magenta")
    title.append("██║  ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║██║ ╚████║\n", style="bold green")
    title.append("╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝\n", style="bold green")
    title.append("\nBENCHMARK   •   HUMANNESS   •   v", style="bold white")
    title.append(VERSION, style="bold green")

    return Panel(
        Align.center(title),
        border_style="bright_cyan",
        box=box.DOUBLE,
        padding=(1, 2),
    )


def render_config(
    model_backend_config: BackendConfig,
    model: str,
    judge_backend_config: BackendConfig,
    judge_config: JudgeConfig,
    prompts_path: Path,
    output_path: Path,
    prompt_count: int,
) -> Group:
    judge_provider_label = (
        judge_config.provider_label
        if judge_backend_config.name == OPENROUTER_BACKEND
        else f"n/a ({judge_backend_config.label} native)"
    )
    return Group(
        Text(""),
        render_kv_line("Tested model", model),
        render_kv_line("Judge model", judge_config.model),
        render_kv_line("Tested provider", model_backend_config.label),
        render_kv_line("Judge provider", judge_backend_config.label),
        render_kv_line("Judge reasoning", judge_config.reasoning_label),
        render_kv_line("Judge routing", judge_provider_label),
        render_kv_line("Prompts", str(prompt_count)),
        render_separator(),
        Text(""),
    )


def render_kv_line(label: str, value: str) -> Text:
    line = Text("  ")
    line.append(f"{label:<20}", style="bold cyan")
    line.append(": ", style="bright_black")
    line.append(value, style="white")
    return line


def render_separator(width: int = 58) -> Text:
    line = Text("  ")
    line.append("─" * width, style="bright_black")
    return line


def render_verbose_answer(item: PromptItem, response_text: str) -> Panel:
    body = Text(response_text, style="white")
    return Panel(
        body,
        title=f"[bold cyan]{item.id}[/bold cyan] model response",
        border_style="bright_black",
        box=box.ROUNDED,
    )


def render_result_row(item: PromptItem, score: Score) -> Table:
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("id", no_wrap=True, style="bold cyan")
    table.add_column("status", no_wrap=True)
    table.add_column("score", no_wrap=True, justify="right")
    table.add_column("bar", no_wrap=True)
    table.add_column("verdict", overflow="fold")

    color = score_color(score.total)
    status_label = "×" if score.total == 0 else "✓"
    status = Text(status_label, style=f"bold {color}")
    table.add_row(
        item.id,
        status,
        Text(f"{score.total}/100", style=f"bold {color}"),
        render_score_bar(score.total, width=24),
        Text(score.verdict or score_label(score.total), style="white"),
    )
    return table


def render_score_bar(score: int, *, width: int = 24) -> Text:
    ratio = min(1.0, max(0.0, score / 100))
    filled = int(round(width * ratio))
    return render_block_bar(
        filled=filled,
        width=width,
        complete_style=score_color(score),
        remaining_style="grey23",
    )


def render_block_bar(
    *,
    filled: int,
    width: int,
    complete_style: str,
    remaining_style: str,
) -> Text:
    filled = max(0, min(width, filled))
    bar = Text()
    bar.append("[", style="bright_black")
    bar.append("█" * filled, style=complete_style)
    bar.append("░" * (width - filled), style=remaining_style)
    bar.append("]", style="bright_black")
    return bar


def render_final_summary(report: dict[str, Any], output_path: Path) -> Group:
    summary = report["summary"]
    final_score = int(summary["score_final"])
    color = score_color(final_score)

    score_line = Text("  ")
    score_line.append("FINAL SCORE     ", style="bold cyan")
    score_line.append(f"{final_score:>3}%   ", style=f"bold {color}")
    score_line.append(score_indicator(final_score), style=f"bold {color}")
    score_line.append(f" {summary['interpretation']}", style=f"bold {color}")

    return Group(
        Text(""),
        render_separator(),
        score_line,
        render_kv_line("Format", f"{summary['format']}/33"),
        render_kv_line("Density", f"{summary['densite']}/33"),
        render_kv_line("Tone", f"{summary['ton']}/34"),
        render_separator(),
        render_kv_line("Report saved", str(output_path)),
    )


def score_indicator(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 60:
        return "🟡"
    if score >= 40:
        return "🟠"
    return "🔴"


def score_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 60:
        return "yellow"
    if score >= 40:
        return "orange3"
    return "red"


def score_label(score: int) -> str:
    if score >= 80:
        return "Very human"
    if score >= 60:
        return "Good, with visible AI tics"
    if score >= 40:
        return "Clearly AI-styled"
    return "Sedated PowerPoint"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_int(value: Any, low: int, high: int, label: str) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"Invalid judge score for {label}: {value!r}") from exc
    return max(low, min(high, number))


if __name__ == "__main__":
    raise SystemExit(main())
