"""
HumanBench CLI
==============
Entry point for the `humanbench` command.

  humanbench config            Interactive setup wizard (run any time)
  humanbench run [model]       Run the benchmark
  humanbench --help / --version
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
from rich.rule import Rule
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

# ── Project root resolution ────────────────────────────────────────────────
# Works for `pip install -e .` (editable): __file__ is the real source file.
# Falls back to CWD for direct invocation.

def _find_project_root() -> Path:
    candidate = Path(__file__).parent.parent  # humanbench/ -> HumanBench/
    if (candidate / "prompts.json").exists():
        return candidate
    return Path.cwd()

_PROJECT_ROOT = _find_project_root()
_ENV_PATH = _PROJECT_ROOT / ".env"

# ── Constants ──────────────────────────────────────────────────────────────

VERSION = "0.1.0"
DEFAULT_PROMPTS_PATH      = _PROJECT_ROOT / "prompts.json"
DEFAULT_JUDGE_PROMPT_PATH = _PROJECT_ROOT / "judge" / "prompt.txt"
DEFAULT_JUDGE_MODEL       = "deepseek/deepseek-v4-pro"
DEFAULT_PROVIDER          = "auto"
TESTED_MODEL_MAX_TOKENS   = 900
DEFAULT_JUDGE_MAX_TOKENS  = 8192
MODEL_GENERATION_ATTEMPTS = 3
JUDGE_GENERATION_ATTEMPTS = 3
AUTO_PUSH_ENV             = "HUMANBENCH_AUTO_PUSH"
DEEPSEEK_V4_PRO_MODEL     = "deepseek/deepseek-v4-pro"
DEEPSEEK_OFFICIAL_PROVIDER = "deepseek"
OPENROUTER_BACKEND = "openrouter"
DEEPSEEK_BACKEND   = "deepseek"
ANTHROPIC_BACKEND  = "anthropic"
OPENAI_BACKEND     = "openai"
MISTRAL_BACKEND    = "mistral"
GOOGLE_BACKEND     = "google"
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
    ("custom", "Other / Custom", None, []),
]


# ── Domain exceptions / helpers ────────────────────────────────────────────

class BenchmarkError(RuntimeError):
    pass


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


# ── Data classes ───────────────────────────────────────────────────────────

class BlockBarColumn(ProgressColumn):
    def __init__(self, *, width: int = 32, complete_style: str = "green", remaining_style: str = "grey23") -> None:
        super().__init__()
        self.width = width
        self.complete_style = complete_style
        self.remaining_style = remaining_style

    def render(self, task: Any) -> Text:
        total = float(task.total or 0)
        completed = float(task.completed or 0)
        ratio = 0.0 if total <= 0 else min(1.0, max(0.0, completed / total))
        filled = int(round(self.width * ratio))
        return render_block_bar(filled=filled, width=self.width, complete_style=self.complete_style, remaining_style=self.remaining_style)


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
            raise BenchmarkError(f"Prompt #{index + 1}: missing required field(s): {', '.join(missing)}")
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
        return {"model": self.model, "max_tokens": self.max_tokens, "reasoning": self.reasoning, "provider": self.provider}


@dataclass(frozen=True)
class BackendConfig:
    name: str
    api_key: str
    api_key_source: str

    @property
    def label(self) -> str:
        return {
            OPENROUTER_BACKEND: "OpenRouter",
            DEEPSEEK_BACKEND:   "DeepSeek",
            ANTHROPIC_BACKEND:  "Anthropic",
            OPENAI_BACKEND:     "OpenAI",
            MISTRAL_BACKEND:    "Mistral",
            GOOGLE_BACKEND:     "Google Gemini",
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
        format_score  = _clamp_int(payload.get("format"), 0, 33, "format")
        densite_score = _clamp_int(payload.get("densite", payload.get("densité")), 0, 33, "densite")
        ton_score     = _clamp_int(payload.get("ton"), 0, 34, "ton")
        return cls(
            format=format_score,
            densite=densite_score,
            ton=ton_score,
            total=format_score + densite_score + ton_score,
            commentaire_format=str(payload.get("commentaire_format", "")).strip(),
            commentaire_densite=str(payload.get("commentaire_densite", "")).strip(),
            commentaire_ton=str(payload.get("commentaire_ton", "")).strip(),
            verdict=str(payload.get("verdict", "")).strip(),
        )

    @classmethod
    def empty_model_response(cls) -> "Score":
        return cls(
            format=0, densite=0, ton=0, total=0,
            commentaire_format="No response to evaluate.",
            commentaire_densite="No measurable density.",
            commentaire_ton="No observable tone.",
            verdict="The tested model returned no text.",
        )


class RichArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args: Any, console: Console, help_fn: Any = None, **kwargs: Any) -> None:
        self.console = console
        self.help_fn = help_fn
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        self.console.print(Panel(Text(message, style="bold red"), title="[bold red]Invalid argument[/bold red]", border_style="red", box=box.ROUNDED))
        if self.help_fn:
            self.help_fn(self.console)
        raise SystemExit(2)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    configure_terminal_encoding()
    console = Console()
    argv = sys.argv[1:]

    if not argv:
        _print_top_help(console)
        return 0

    subcmd = argv[0]
    rest   = argv[1:]

    if subcmd in ("-h", "--help"):
        _print_top_help(console)
        return 0

    if subcmd in ("-V", "--version"):
        console.print(f"humanbench {VERSION}")
        return 0

    if subcmd == "config":
        if any(a in ("-h", "--help") for a in rest):
            _print_config_help(console)
            return 0
        return _cmd_config(console)

    if subcmd == "run":
        return _cmd_run(console, rest)

    # ── Backwards compat ──────────────────────────────────────────────────
    # humanbench --model X  |  humanbench anthropic/claude-sonnet-4-6
    return _cmd_run(console, argv)


def _cmd_config(console: Console) -> int:
    """humanbench config — always run the wizard, even if .env already exists."""
    dummy_args = argparse.Namespace(model=None, provider=None, prompts=str(DEFAULT_PROMPTS_PATH), verbose=False, output=None)
    try:
        run_setup_wizard(console, dummy_args, reconfigure=True)
    except KeyboardInterrupt:
        console.print()
        console.print(Panel(Text("Setup cancelled.", style="bold red"), title="[bold red]Cancelled[/bold red]", border_style="red", box=box.ROUNDED))
        return 130
    return 0


def _cmd_run(console: Console, argv: list[str]) -> int:
    """humanbench run [model ...] [options] — run the benchmark on one or more models."""
    # Collect leading positional model names (stop at first flag)
    models_positional: list[str] = []
    rest = list(argv)
    while rest and not rest[0].startswith("-"):
        models_positional.append(rest.pop(0))

    if any(a in ("-h", "--help") for a in rest):
        _print_run_help(console)
        return 0

    parser = _build_run_parser(console)
    args = parser.parse_args(rest)

    # Combine positional models with --model flag (deduped, positional first)
    flag_model = args.model or ""
    all_models: list[str] = list(dict.fromkeys(
        m for m in models_positional + ([flag_model] if flag_model else []) if m
    ))

    if _wizard_needed():
        try:
            run_setup_wizard(console, args, reconfigure=False)
        except KeyboardInterrupt:
            console.print()
            console.print(Panel(Text("Setup cancelled.", style="bold red"), title="[bold red]Cancelled[/bold red]", border_style="red", box=box.ROUNDED))
            return 130

    runs_per_model = max(1, int(getattr(args, "runs", 1) or 1))

    # Single model + single run → original behaviour
    if len(all_models) <= 1 and runs_per_model == 1:
        if all_models:
            args.model = all_models[0]
        try:
            run(args, console)
        except KeyboardInterrupt:
            console.print(Panel("Benchmark interrupted.", title="[bold yellow]Stop[/bold yellow]", border_style="yellow", box=box.ROUNDED))
            return 130
        except (BenchmarkError, AdapterError) as exc:
            console.print(Panel(str(exc), title="[bold red]Benchmark blocked[/bold red]", border_style="red", box=box.ROUNDED))
            return 1
        return 0

    if not all_models:
        # No models given but --runs > 1: fall back to MODEL_TESTED
        env_model = _optional_string(os.getenv("MODEL_TESTED"))
        if not env_model:
            console.print(Panel("No model specified. Use `humanbench run <model>` or set MODEL_TESTED in .env.", title="[bold red]Error[/bold red]", border_style="red", box=box.ROUNDED))
            return 1
        all_models = [env_model]

    # ── Multi-model and/or multi-run flow ────────────────────────────────────
    aggregated_per_model: list[dict[str, Any]] = []
    failed:               list[str]            = []

    for mi, model in enumerate(all_models):
        if mi > 0:
            console.print()
            console.print(Rule(style="bright_black"))
            console.print()

        args.model = model

        if runs_per_model == 1:
            try:
                report = run(args, console)
                aggregated_per_model.append(report)
            except KeyboardInterrupt:
                console.print(Panel("Benchmark interrupted.", title="[bold yellow]Stop[/bold yellow]", border_style="yellow", box=box.ROUNDED))
                break
            except (BenchmarkError, AdapterError) as exc:
                console.print(Panel(f"[bold]{model}[/bold]\n{exc}", title="[bold red]Skipped[/bold red]", border_style="red", box=box.ROUNDED))
                failed.append(model)
            continue

        # runs_per_model > 1 → run N times silently w.r.t. leaderboard, then aggregate
        single_runs: list[dict[str, Any]] = []
        args._skip_publish = True
        try:
            for ri in range(runs_per_model):
                console.print()
                console.print(Panel(f"[bold cyan]{model}[/bold cyan]   run [bold]{ri + 1}[/bold] / {runs_per_model}", border_style="bright_black", box=box.ROUNDED, padding=(0, 2)))
                report = run(args, console)
                single_runs.append(report)
        except KeyboardInterrupt:
            console.print(Panel("Benchmark interrupted.", title="[bold yellow]Stop[/bold yellow]", border_style="yellow", box=box.ROUNDED))
            break
        except (BenchmarkError, AdapterError) as exc:
            console.print(Panel(f"[bold]{model}[/bold]\n{exc}", title="[bold red]Skipped[/bold red]", border_style="red", box=box.ROUNDED))
            failed.append(model)
            args._skip_publish = False
            continue
        args._skip_publish = False

        if not single_runs:
            continue

        aggregated = aggregate_runs(single_runs)
        agg_path = aggregated_output_path(aggregated.get("requested_model") or aggregated.get("model") or "model", len(single_runs))
        aggregated["output_path"] = str(agg_path)
        write_report(agg_path, aggregated)

        console.print(render_aggregated_summary(aggregated, agg_path))

        # One leaderboard entry per model (the aggregate)
        sync_leaderboard(console, aggregated)
        publish_leaderboard_if_enabled(console, args, aggregated)

        aggregated_per_model.append(aggregated)

    if len(aggregated_per_model) > 1:
        console.print()
        console.print(Rule(style="bright_cyan"))
        console.print(render_multi_comparison(aggregated_per_model))

    if failed:
        console.print(Panel("\n".join(f"  • {m}" for m in failed), title="[bold red]Failed models[/bold red]", border_style="red", box=box.ROUNDED))

    return 0


# ── Help printers ──────────────────────────────────────────────────────────

def _print_top_help(console: Console) -> None:
    cmds = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", border_style="bright_black")
    cmds.add_column("Command", style="cyan", no_wrap=True)
    cmds.add_column("Description", style="white")
    cmds.add_row("humanbench config",              "Interactive setup wizard — configure keys & default model")
    cmds.add_row("humanbench run [model ...]",     "Run the benchmark (one or more models)")
    cmds.add_row("humanbench run --help",          "Show all options for the run subcommand")
    cmds.add_row("humanbench --version",     "Show version")
    cmds.add_row("humanbench --help",        "Show this message")

    console.print(render_banner())
    console.print(Panel(cmds, title="[bold cyan]humanbench[/bold cyan]", border_style="cyan", box=box.ROUNDED))


def _print_run_help(console: Console) -> None:
    usage = Table.grid(padding=(0, 1))
    usage.add_column(style="bold cyan", no_wrap=True)
    usage.add_column(style="white")
    usage.add_row("Usage",   "humanbench run [model] [options]")
    usage.add_row("",        "humanbench run anthropic/claude-sonnet-4-6")
    usage.add_row("",        "humanbench run anthropic/claude-opus-4-7 -n 5      [bold green]# 5 runs averaged[/bold green]")
    usage.add_row("",        "humanbench run claude-sonnet-4-6 gpt-4.1 -n 3      [bold green]# 3 runs × 2 models[/bold green]")
    usage.add_row("",        "humanbench run --model openai/gpt-4.1 --verbose")

    opts = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", border_style="bright_black")
    opts.add_column("Option",      style="cyan", no_wrap=True)
    opts.add_column("Description", style="white")
    opts.add_row("[model ...]",    "One or more models to test (positional).")
    opts.add_row("--model",        "Model to test (flag, backwards compatible).")
    opts.add_row("--runs, -n N",   "Number of runs per model to average scores (default 1).")
    opts.add_row("--provider",     "API backend: auto, anthropic, openai, mistral, google, deepseek, openrouter.")
    opts.add_row("--prompts",      f"Custom prompt JSON file. Default: {DEFAULT_PROMPTS_PATH.name}")
    opts.add_row("--verbose",      "Print each full model response in real time.")
    opts.add_row("--output",       "JSON report path. Default: results/<model>_<date>.json")
    opts.add_row("--no-push",      "Skip automatic git commit/push after syncing the leaderboard.")
    opts.add_row("-h, --help",     "Show this help.")

    console.print(render_banner())
    console.print(Panel(Group(usage, Text(""), opts), title="[bold cyan]humanbench run[/bold cyan]", border_style="cyan", box=box.ROUNDED))


def _print_config_help(console: Console) -> None:
    body = Text.assemble(
        ("humanbench config\n\n", "bold cyan"),
        ("Launches the interactive setup wizard.\n", "white"),
        ("Can be run at any time to update API keys or the default model.\n", "white"),
        ("Saves configuration to ", "white"),
        (str(_ENV_PATH), "bold cyan"),
        (".", "white"),
    )
    console.print(render_banner())
    console.print(Panel(body, title="[bold cyan]humanbench config[/bold cyan]", border_style="cyan", box=box.ROUNDED))


# ── Encoding + parser ──────────────────────────────────────────────────────

def configure_terminal_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _build_run_parser(console: Console) -> RichArgumentParser:
    parser = RichArgumentParser(prog="humanbench run", add_help=False, console=console, help_fn=_print_run_help)
    parser.add_argument("--model",    default=None,                       help="Model to test.")
    parser.add_argument("--provider", default=None, choices=sorted(SUPPORTED_BACKENDS), help="API backend.")
    parser.add_argument("--prompts",  default=str(DEFAULT_PROMPTS_PATH),  help="Path to the prompt JSON file.")
    parser.add_argument("--verbose",  action="store_true",                help="Print full model responses.")
    parser.add_argument("--output",   default=None,                       help="JSON report output path.")
    parser.add_argument("--no-push",  action="store_true",                help="Skip automatic git commit/push after syncing the leaderboard.")
    parser.add_argument("--runs", "-n", type=int, default=1, metavar="N",  help="Number of runs per model to average scores (default 1).")
    return parser


# Alias kept for backwards compatibility with run_benchmark.py
def build_parser(console: Console) -> RichArgumentParser:
    return _build_run_parser(console)


# ═══════════════════════════════════════════════════════════════════════════
#  SETUP WIZARD
# ═══════════════════════════════════════════════════════════════════════════

def _wizard_needed() -> bool:
    if not _ENV_PATH.exists():
        return True
    values = dotenv_values(str(_ENV_PATH))
    key = (values.get("OPENROUTER_API_KEY") or "").strip()
    return not key or key == "sk-or-xxxx"


def _wizard_menu_table(items: list[tuple[str, str]]) -> Table:
    t = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    t.add_column(style="bold green", no_wrap=True)
    t.add_column(style="white")
    for num, label in items:
        t.add_row(num, label)
    return t


def _wizard_pick(console: Console, n: int) -> int:
    while True:
        raw = Prompt.ask(f"  [cyan]Choice [bold][1-{n}][/bold][/cyan]", console=console)
        try:
            choice = int(raw.strip())
            if 1 <= choice <= n:
                return choice - 1
        except ValueError:
            pass
        console.print(f"  [bold red]Enter a number between 1 and {n}.[/bold red]")


def _wizard_input(console: Console, prompt: str, *, secret: bool = False) -> str:
    while True:
        val = Prompt.ask(f"  [cyan]{prompt}[/cyan]", password=secret, console=console).strip()
        if val:
            return val
        console.print("  [bold red]This field is required.[/bold red]")


def _wizard_mask(key: str) -> str:
    return ("*" * 8) + key[-4:] if len(key) >= 4 else "****"


def run_setup_wizard(
    console: Console,
    args: argparse.Namespace,
    *,
    reconfigure: bool = False,
) -> None:
    S = "bold cyan"

    console.print(render_banner())

    if reconfigure:
        welcome = Text.assemble(
            ("RECONFIGURATION\n\n", "bold green"),
            ("Update your API keys or default model.\n", "dim white"),
            ("File ", "white"),
            (str(_ENV_PATH), "bold cyan"),
            (" will be overwritten.", "white"),
        )
        confirm_prompt = "  [cyan]Save this configuration?[/cyan]"
        done_msg = Text.assemble((".env updated  ✓", "bold green"))
    else:
        welcome = Text.assemble(
            ("FIRST LAUNCH DETECTED\n\n", "bold green"),
            (".env missing or OPENROUTER_API_KEY not set.\n", "dim white"),
            ("This wizard configures your environment in ", "white"),
            ("30 seconds", "bold cyan"),
            (" and launches the benchmark immediately.", "white"),
        )
        confirm_prompt = "  [cyan]Save and run the benchmark?[/cyan]"
        done_msg = Text.assemble((".env saved  ✓\n\n", "bold green"), ("Launching benchmark...", "dim white"))

    console.print(Panel(welcome, title="[bold green]// SETUP WIZARD[/bold green]", border_style="green", box=box.DOUBLE, padding=(1, 2)))

    # ── Step 1: provider ──────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        _wizard_menu_table([(f"[{i + 1}]", name) for i, (_, name, _, _) in enumerate(_WIZARD_PROVIDERS)]),
        title=f"[{S}]STEP 1 / 5  ·  Model provider[/{S}]",
        border_style="cyan", box=box.ROUNDED, padding=(1, 2),
    ))
    provider_idx = _wizard_pick(console, len(_WIZARD_PROVIDERS))
    provider_id, provider_name, key_env_name, provider_models = _WIZARD_PROVIDERS[provider_idx]

    if provider_id == "custom":
        key_env_name = _wizard_input(console, "Environment variable name (e.g. MY_API_KEY)")

    # ── Step 2: provider API key ──────────────────────────────────────────
    console.print()
    console.print(Panel(
        Text(f"Copy your {provider_name} API key from your dashboard.", style="white"),
        title=f"[{S}]STEP 2 / 5  ·  API Key — {provider_name}[/{S}]",
        border_style="cyan", box=box.ROUNDED, padding=(1, 2),
    ))
    provider_api_key = _wizard_input(console, key_env_name or "API_KEY", secret=True)

    # ── Step 3: model ─────────────────────────────────────────────────────
    console.print()
    if provider_models:
        model_options = provider_models + ["Custom model — enter manually"]
        console.print(Panel(
            _wizard_menu_table([(f"[{i + 1}]", m) for i, m in enumerate(model_options)]),
            title=f"[{S}]STEP 3 / 5  ·  Model ({provider_name})[/{S}]",
            border_style="cyan", box=box.ROUNDED, padding=(1, 2),
        ))
        model_idx = _wizard_pick(console, len(model_options))
        chosen_model = _wizard_input(console, "Model name") if model_idx == len(model_options) - 1 else provider_models[model_idx]
    else:
        console.print(Panel(
            Text("Enter the full model name (e.g. org/model-name).", style="white"),
            title=f"[{S}]STEP 3 / 5  ·  Model[/{S}]",
            border_style="cyan", box=box.ROUNDED, padding=(1, 2),
        ))
        chosen_model = _wizard_input(console, "Model name")

    # ── Step 4: OpenRouter key (scoring) ─────────────────────────────────
    console.print()
    if provider_id == OPENROUTER_BACKEND:
        openrouter_key = provider_api_key
        console.print(Panel(
            Text("OpenRouter is your main provider → key reused for scoring.", style="green"),
            title=f"[{S}]STEP 4 / 5  ·  OpenRouter key (scoring)[/{S}]",
            border_style="cyan", box=box.ROUNDED, padding=(1, 1),
        ))
    else:
        console.print(Panel(
            Text.assemble(
                ("Scoring runs via OpenRouter.\n", "white"),
                ("Create a free account at openrouter.ai if needed.", "white"),
            ),
            title=f"[{S}]STEP 4 / 5  ·  OpenRouter key (scoring)[/{S}]",
            border_style="cyan", box=box.ROUNDED, padding=(1, 2),
        ))
        openrouter_key = _wizard_input(console, "OPENROUTER_API_KEY", secret=True)

    # ── Step 5: summary + confirmation ───────────────────────────────────
    console.print()
    recap = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    recap.add_column(style="bold cyan", no_wrap=True)
    recap.add_column(style="white")
    recap.add_row("Provider", provider_name)
    recap.add_row("Model",    chosen_model)
    recap.add_row(key_env_name or "API_KEY", _wizard_mask(provider_api_key))
    if provider_id != OPENROUTER_BACKEND:
        recap.add_row("OPENROUTER_API_KEY", _wizard_mask(openrouter_key))

    console.print(Panel(recap, title="[bold green]STEP 5 / 5  ·  Summary[/bold green]", border_style="green", box=box.DOUBLE, padding=(1, 1)))
    console.print()

    confirmed = Confirm.ask(confirm_prompt, default=True, console=console)
    if not confirmed:
        console.print()
        console.print(Panel(Text("Setup cancelled. No files modified.", style="bold red"), title="[bold red]Cancelled[/bold red]", border_style="red", box=box.ROUNDED))
        raise SystemExit(0)

    # ── Write .env ────────────────────────────────────────────────────────
    env_backend = provider_id if provider_id != "custom" else DEFAULT_PROVIDER
    lines: list[str] = [
        f"PROVIDER={env_backend}",
        f"MODEL_TESTED={chosen_model}",
        f"OPENROUTER_API_KEY={openrouter_key}",
    ]
    if key_env_name and provider_id != OPENROUTER_BACKEND:
        lines.append(f"{key_env_name}={provider_api_key}")

    _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Propagate immediately into os.environ
    os.environ["PROVIDER"] = env_backend
    os.environ["MODEL_TESTED"] = chosen_model
    os.environ["OPENROUTER_API_KEY"] = openrouter_key
    if key_env_name:
        os.environ[key_env_name] = provider_api_key

    if not reconfigure:
        if not args.model:
            args.model = chosen_model
        if not args.provider and provider_id in SUPPORTED_BACKENDS:
            args.provider = provider_id

    console.print()
    console.print(Panel(done_msg, border_style="green", box=box.ROUNDED, padding=(0, 2)))
    console.print()


# ═══════════════════════════════════════════════════════════════════════════
#  BENCHMARK CORE
# ═══════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace, console: Console) -> dict[str, Any]:
    load_dotenv(str(_ENV_PATH))

    requested_model = _optional_string(args.model) or _optional_string(os.getenv("MODEL_TESTED"))
    if not requested_model:
        raise BenchmarkError("No tested model configured. Use `humanbench run <model>` or set MODEL_TESTED in .env.")

    judge_model = DEFAULT_JUDGE_MODEL
    model_backend_config = build_backend_config(
        requested_provider=_optional_string(args.provider) or _optional_string(os.getenv("PROVIDER")),
        model=requested_model,
        role="tested model",
    )
    judge_backend_config = build_backend_config(requested_provider=None, model=judge_model, role="judge")
    backend_model       = normalize_model_for_backend(requested_model, model_backend_config.name)
    backend_judge_model = normalize_model_for_backend(judge_model, judge_backend_config.name)
    judge_config  = build_judge_config(backend_judge_model, judge_backend_config.name)
    prompts_path  = Path(args.prompts)
    output_path   = Path(args.output) if args.output else default_output_path(backend_model)

    prompts      = load_prompts(prompts_path)
    judge_prompt = load_text(DEFAULT_JUDGE_PROMPT_PATH, "judge prompt")

    console.print(render_banner())
    console.print(render_config(model_backend_config, backend_model, judge_backend_config, judge_config, prompts_path, output_path, len(prompts)))

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
                response_text    = model_completion.content

                if args.verbose:
                    console.print(render_verbose_answer(item, response_text))

                if not response_text.strip():
                    score  = Score.empty_model_response()
                    result = build_prompt_result(
                        item, response_text, score,
                        raw_judge_text="", raw_judge_json={},
                        model_raw_json=model_completion.raw,
                        error={"stage": "model", "type": "empty_response", "message": "The tested model returned no text after multiple attempts."},
                    )
                    results.append(result)
                    console.print(render_result_row(item, score))
                    progress.advance(task_id)
                    continue

                score, raw_judge_text, raw_judge_json = judge_answer(judge_adapter, judge_config, judge_prompt, item, response_text)
                result = build_prompt_result(item, response_text, score, raw_judge_text, raw_judge_json, model_raw_json=model_completion.raw)
                results.append(result)
                console.print(render_result_row(item, score))
                progress.advance(task_id)

    report = build_report(model_backend_config, judge_backend_config, requested_model, backend_model, backend_judge_model, judge_config, prompts_path, output_path, results)
    write_report(output_path, report)
    console.print(render_final_summary(report, output_path))
    if not getattr(args, "_skip_publish", False):
        sync_leaderboard(console, report)
        publish_leaderboard_if_enabled(console, args, report)
    return report


# ── Backend / adapter helpers ──────────────────────────────────────────────

def build_backend_config(*, requested_provider: str | None, model: str, role: str, fallback_provider: str | None = None) -> BackendConfig:
    requested_provider = (requested_provider or DEFAULT_PROVIDER).strip().lower()
    if requested_provider not in SUPPORTED_BACKENDS:
        raise BenchmarkError(f"Invalid PROVIDER for {role}: {requested_provider!r}. Use auto, openrouter, anthropic, openai, mistral, google, or deepseek.")

    if requested_provider == DEFAULT_PROVIDER:
        inferred = infer_backend_for_model(model)
        if inferred is not None:
            requested_provider = inferred
        elif fallback_provider is not None:
            requested_provider = fallback_provider
        else:
            requested_provider = _infer_backend_from_keys()
            if requested_provider is None:
                raise BenchmarkError(f"Could not infer a backend for {role}. Set PROVIDER or add a provider API key.")

    api_key, source = resolve_api_key(requested_provider)
    if not api_key:
        env_name = preferred_key_name(requested_provider)
        raise BenchmarkError(f"{env_name} is missing for {role}. Add it to .env or use API_KEY with an explicit PROVIDER.")
    return BackendConfig(name=requested_provider, api_key=api_key, api_key_source=source)


def build_adapter(backend_config: BackendConfig) -> ModelAdapter:
    if backend_config.name == OPENROUTER_BACKEND: return OpenRouterAdapter(backend_config.api_key)
    if backend_config.name == DEEPSEEK_BACKEND:   return DeepSeekAdapter(backend_config.api_key)
    if backend_config.name == ANTHROPIC_BACKEND:  return AnthropicAdapter(backend_config.api_key)
    if backend_config.name == OPENAI_BACKEND:     return OpenAIAdapter(backend_config.api_key)
    if backend_config.name == MISTRAL_BACKEND:    return MistralAdapter(backend_config.api_key)
    if backend_config.name == GOOGLE_BACKEND:     return GoogleAdapter(backend_config.api_key)
    raise BenchmarkError(f"Unsupported backend: {backend_config.name!r}")


def infer_backend_for_model(model: str) -> str | None:
    model = model.strip().lower()
    if not model: return None
    if model.startswith("deepseek/"): return OPENROUTER_BACKEND
    if model.startswith(("anthropic/", "openai/", "mistralai/", "google/")): return OPENROUTER_BACKEND
    if model in {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"}: return DEEPSEEK_BACKEND
    if model.startswith("claude-"): return ANTHROPIC_BACKEND
    if model.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt-")): return OPENAI_BACKEND
    if model.startswith(("mistral-", "ministral-", "codestral-", "magistral-")): return MISTRAL_BACKEND
    if model.startswith(("gemini-", "models/gemini-")): return GOOGLE_BACKEND
    if "/" in model: return OPENROUTER_BACKEND
    return None


def _infer_backend_from_keys() -> str | None:
    for provider in (OPENROUTER_BACKEND, DEEPSEEK_BACKEND, ANTHROPIC_BACKEND, OPENAI_BACKEND, MISTRAL_BACKEND, GOOGLE_BACKEND):
        api_key, _ = resolve_api_key(provider, allow_generic=False)
        if api_key:
            return provider
    return None


def resolve_api_key(provider: str, *, allow_generic: bool = True) -> tuple[str | None, str]:
    env_names = {
        OPENROUTER_BACKEND: ("OPENROUTER_API_KEY",),
        DEEPSEEK_BACKEND:   ("DEEPSEEK_API_KEY",),
        ANTHROPIC_BACKEND:  ("ANTHROPIC_API_KEY",),
        OPENAI_BACKEND:     ("OPENAI_API_KEY",),
        MISTRAL_BACKEND:    ("MISTRAL_API_KEY",),
        GOOGLE_BACKEND:     ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
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
        DEEPSEEK_BACKEND:   "DEEPSEEK_API_KEY",
        ANTHROPIC_BACKEND:  "ANTHROPIC_API_KEY",
        OPENAI_BACKEND:     "OPENAI_API_KEY",
        MISTRAL_BACKEND:    "MISTRAL_API_KEY",
        GOOGLE_BACKEND:     "GOOGLE_API_KEY",
    }.get(provider, "API_KEY")


def normalize_model_for_backend(model: str, backend_name: str) -> str:
    model = model.strip()
    if backend_name == OPENROUTER_BACKEND:
        if model in {"deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"}:
            return f"deepseek/{model}"
        return model
    if backend_name == DEEPSEEK_BACKEND:
        if model.startswith("deepseek/"): return model.split("/", 1)[1]
        if "/" in model: raise BenchmarkError("The DeepSeek backend expects native model names like `deepseek-v4-pro`, not OpenRouter slugs.")
        return model
    if backend_name == ANTHROPIC_BACKEND:
        if model.startswith("anthropic/"): return model.split("/", 1)[1]
        if "/" in model: raise BenchmarkError("The Anthropic backend expects a native model name like `claude-sonnet-4-5`, not an OpenRouter slug.")
        return model
    if backend_name == OPENAI_BACKEND:
        if model.startswith("openai/"): return model.split("/", 1)[1]
        if "/" in model: raise BenchmarkError("The OpenAI backend expects a native model name like `gpt-4.1`, not an OpenRouter slug.")
        return model
    if backend_name == MISTRAL_BACKEND:
        if model.startswith(("mistralai/", "mistral/")): return model.split("/", 1)[1]
        if "/" in model: raise BenchmarkError("The Mistral backend expects a native model name like `mistral-large-latest`, not an OpenRouter slug.")
        return model
    if backend_name == GOOGLE_BACKEND:
        if model.startswith("google/"): return model.split("/", 1)[1]
        if "/" in model and not model.startswith("models/"): raise BenchmarkError("The Google backend expects a native model name like `gemini-2.5-flash`, not an OpenRouter slug.")
        return model
    return model


# ── Generation ─────────────────────────────────────────────────────────────

def generate_answer(adapter: ModelAdapter, model: str, item: PromptItem) -> Generation:
    last_completion: Generation | None = None
    last_error: AdapterError | None = None
    for attempt in range(1, MODEL_GENERATION_ATTEMPTS + 1):
        try:
            completion = adapter.generate(
                [{"role": "user", "content": item.prompt}],
                model=model, temperature=0.7, max_tokens=TESTED_MODEL_MAX_TOKENS, allow_empty=True,
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
    normalized_judge   = judge_model.lower()
    default_deepseek_v4 = normalized_judge in {DEEPSEEK_V4_PRO_MODEL, "deepseek-v4-pro"}
    max_tokens = DEFAULT_JUDGE_MAX_TOKENS
    if max_tokens < 256:
        raise BenchmarkError("DEFAULT_JUDGE_MAX_TOKENS must be >= 256. A judge with no budget is just vibes in a robe.")
    effort = "xhigh" if default_deepseek_v4 else None
    reasoning: dict[str, Any] | None = {"effort": effort, "exclude": True} if effort and effort.lower() != "auto" else None
    provider: dict[str, Any] | None = None
    if backend_name == OPENROUTER_BACKEND:
        provider_order = [DEEPSEEK_OFFICIAL_PROVIDER] if default_deepseek_v4 else []
        if provider_order:
            provider = {"order": provider_order, "allow_fallbacks": False, "require_parameters": True}
    return JudgeConfig(model=judge_model, max_tokens=max_tokens, reasoning=reasoning, provider=provider)


def judge_answer(adapter: ModelAdapter, judge_config: JudgeConfig, judge_prompt: str, item: PromptItem, response_text: str) -> tuple[Score, str, dict[str, Any]]:
    messages = build_judge_messages(judge_prompt, item, response_text)
    last_raw_text = ""
    last_error: Exception | None = None
    for attempt in range(JUDGE_GENERATION_ATTEMPTS):
        completion = adapter.generate(
            messages, model=judge_config.model, temperature=0,
            max_tokens=judge_config.max_tokens, response_format={"type": "json_object"},
            reasoning=judge_config.reasoning, provider=judge_config.provider, allow_empty=True,
        )
        last_raw_text = completion.content
        if not last_raw_text.strip():
            last_error = BenchmarkError("The judge returned no text.")
            if attempt < JUDGE_GENERATION_ATTEMPTS - 1:
                messages.append({"role": "user", "content": "Your previous output was empty. Return only the requested JSON object."})
                time.sleep(0.8 * (attempt + 1))
                continue
            break
        try:
            raw_json = extract_json_object(last_raw_text)
            return Score.from_judge_json(raw_json), last_raw_text, raw_json
        except BenchmarkError as exc:
            last_error = exc
            messages.extend([
                {"role": "assistant", "content": last_raw_text},
                {"role": "user", "content": "Your previous output was not a valid JSON object. Return only the requested JSON object, with no Markdown and no extra commentary."},
            ])
    raise BenchmarkError(f"The judge failed to return valid JSON. Last error: {last_error}")


def build_judge_messages(judge_prompt: str, item: PromptItem, response_text: str) -> list[dict[str, str]]:
    payload = {
        "id": item.id, "type": item.type, "user_prompt": item.prompt,
        "reference_humaine": item.reference_humaine, "reference_score": item.reference_score,
        "notes_juge": item.notes_juge, "ai_response_to_evaluate": response_text,
    }
    return [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": f"Evaluate the AI response below using the HumanBench framework.\nThe context JSON follows:\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"},
    ]


# ── I/O helpers ────────────────────────────────────────────────────────────

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
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start: end + 1])
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


def build_prompt_result(item: PromptItem, response_text: str, score: Score, raw_judge_text: str, raw_judge_json: dict[str, Any], *, model_raw_json: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": item.id, "type": item.type, "prompt": item.prompt, "response": response_text,
        "score":    {"format": score.format, "densite": score.densite, "ton": score.ton, "total": score.total},
        "comments": {"format": score.commentaire_format, "densite": score.commentaire_densite, "ton": score.commentaire_ton, "verdict": score.verdict},
        "reference": {"humaine": item.reference_humaine, "score": item.reference_score, "notes_juge": item.notes_juge},
        "model_raw_json": model_raw_json, "judge_raw_text": raw_judge_text, "judge_raw_json": raw_judge_json,
    }
    if error is not None:
        result["error"] = error
    return result


def build_report(model_backend_config: BackendConfig, judge_backend_config: BackendConfig, requested_model: str, model: str, judge_model: str, judge_config: JudgeConfig, prompts_path: Path, output_path: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "humanbench_version": VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": model_backend_config.name, "judge_backend": judge_backend_config.name,
        "api_key_source": model_backend_config.api_key_source, "judge_api_key_source": judge_backend_config.api_key_source,
        "requested_model": requested_model, "model": model, "judge": judge_model,
        "judge_config": judge_config.to_report(),
        "prompts_path": str(prompts_path), "output_path": str(output_path),
        "summary": summarize_scores(results), "results": results,
    }


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"score_final": 0, "format": 0, "densite": 0, "ton": 0, "interpretation": "No prompts"}
    totals   = [int(r["score"]["total"])   for r in results]
    formats  = [int(r["score"]["format"])  for r in results]
    densites = [int(r["score"]["densite"]) for r in results]
    tons     = [int(r["score"]["ton"])     for r in results]
    final    = round(sum(totals) / len(totals))
    return {
        "score_final": final,
        "format":   round(sum(formats)  / len(formats)),
        "densite":  round(sum(densites) / len(densites)),
        "ton":      round(sum(tons)     / len(tons)),
        "interpretation": score_label(final),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_runs(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an averaged report from N runs of the same model.

    Returns a dict with the same surface as a single report (so the leaderboard
    and comparison table treat it uniformly) plus a `runs` array and per-axis
    standard deviation for variance visibility.
    """
    if not reports:
        raise BenchmarkError("aggregate_runs called with no reports")
    if len(reports) == 1:
        return reports[0]

    base = reports[0]
    finals  = [int(r["summary"]["score_final"]) for r in reports]
    formats = [int(r["summary"]["format"])      for r in reports]
    densis  = [int(r["summary"]["densite"])     for r in reports]
    tons    = [int(r["summary"]["ton"])         for r in reports]

    def _mean(xs: list[int]) -> int:
        return round(sum(xs) / len(xs))

    def _std(xs: list[int]) -> float:
        m = sum(xs) / len(xs)
        return round((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5, 2)

    avg_final = _mean(finals)
    summary = {
        "score_final":   avg_final,
        "format":        _mean(formats),
        "densite":       _mean(densis),
        "ton":           _mean(tons),
        "interpretation": score_label(avg_final),
        "runs_count":    len(reports),
        "stddev": {
            "score_final": _std(finals),
            "format":      _std(formats),
            "densite":     _std(densis),
            "ton":         _std(tons),
        },
        "per_run": [
            {"score_final": finals[i], "format": formats[i], "densite": densis[i], "ton": tons[i]}
            for i in range(len(reports))
        ],
    }

    aggregated = {
        **{k: v for k, v in base.items() if k not in ("summary", "results", "output_path")},
        "created_at":  datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs_count":  len(reports),
        "summary":     summary,
        "runs":        [{"created_at": r["created_at"], "summary": r["summary"], "output_path": r.get("output_path")} for r in reports],
        "results":     reports[-1]["results"],
    }
    return aggregated


def aggregated_output_path(model: str, runs: int) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _PROJECT_ROOT / "results" / f"{safe_model}_{stamp}_avg{runs}.json"


def sync_leaderboard(console: Console, report: dict[str, Any]) -> None:
    results_path      = _PROJECT_ROOT / "results.json"
    site_results_path = _PROJECT_ROOT / "site" / "results.json"
    entry = {"requested_model": report["requested_model"], "created_at": report["created_at"], "summary": report["summary"]}
    leaderboard = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    leaderboard.append(entry)
    results_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    site_results_path.parent.mkdir(parents=True, exist_ok=True)
    site_results_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(Panel(f"Leaderboard synchronized → {site_results_path}", title="[bold green]Leaderboard Synced[/bold green]", border_style="green", box=box.ROUNDED))


def publish_leaderboard_if_enabled(console: Console, args: argparse.Namespace, report: dict[str, Any]) -> None:
    if getattr(args, "no_push", False) or not _auto_push_enabled_from_env():
        console.print(Panel("Auto-push skipped. The leaderboard is local until you push it.", title="[bold yellow]Publish Skipped[/bold yellow]", border_style="yellow", box=box.ROUNDED))
        return
    publish_leaderboard(console, report)


def _auto_push_enabled_from_env() -> bool:
    value = os.getenv(AUTO_PUSH_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def publish_leaderboard(console: Console, report: dict[str, Any]) -> None:
    paths = [_PROJECT_ROOT / "results.json", _PROJECT_ROOT / "site" / "results.json"]
    rel_paths = [path.relative_to(_PROJECT_ROOT).as_posix() for path in paths]

    if not _git_ok(["rev-parse", "--is-inside-work-tree"]):
        console.print(Panel("Auto-push skipped: this project is not inside a git repository.", title="[bold yellow]Publish Skipped[/bold yellow]", border_style="yellow", box=box.ROUNDED))
        return

    if _git_has_changes(rel_paths):
        summary = report["summary"]
        score = int(summary.get("score_final", 0))
        model = str(report.get("requested_model", "model"))
        message = f"Update leaderboard: {model} {score}%"
        commit = _git(["commit", "--only", "-m", message, "--", *rel_paths])
        if commit.returncode != 0:
            console.print(Panel(_git_output(commit), title="[bold red]Publish Failed: Commit[/bold red]", border_style="red", box=box.ROUNDED))
            return

    push_args = _git_push_args()
    if push_args is None:
        console.print(Panel("Auto-push skipped: no current branch or no `origin` remote.", title="[bold yellow]Publish Skipped[/bold yellow]", border_style="yellow", box=box.ROUNDED))
        return

    push = _git(push_args)
    if push.returncode != 0:
        console.print(Panel(_git_output(push), title="[bold red]Publish Failed: Push[/bold red]", border_style="red", box=box.ROUNDED))
        return

    console.print(Panel(_git_output(push) or "Everything up-to-date.", title="[bold green]Leaderboard Published[/bold green]", border_style="green", box=box.ROUNDED))


def _git_has_changes(rel_paths: list[str]) -> bool:
    return _git(["diff", "--quiet", "HEAD", "--", *rel_paths]).returncode != 0


def _git_push_args() -> list[str] | None:
    branch = _git(["branch", "--show-current"])
    current_branch = branch.stdout.strip()
    if branch.returncode != 0 or not current_branch:
        return None

    upstream = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if upstream.returncode == 0:
        return ["push"]

    origin = _git(["remote", "get-url", "origin"])
    if origin.returncode != 0:
        return None
    return ["push", "-u", "origin", current_branch]


def _git_ok(args: list[str]) -> bool:
    return _git(args).returncode == 0


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=_PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_output(result: subprocess.CompletedProcess[str]) -> str:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return output[-3000:] if len(output) > 3000 else output


def default_output_path(model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _PROJECT_ROOT / "results" / f"{safe_model}_{stamp}.json"


# ── Render helpers ─────────────────────────────────────────────────────────

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
    return Panel(Align.center(title), border_style="bright_cyan", box=box.DOUBLE, padding=(1, 2))


def render_config(model_backend_config: BackendConfig, model: str, judge_backend_config: BackendConfig, judge_config: JudgeConfig, prompts_path: Path, output_path: Path, prompt_count: int) -> Group:
    return Group(
        Text(""),
        render_kv_line("Tested model",    model),
        render_kv_line("Tested provider", model_backend_config.label),
        render_kv_line("Prompts",         str(prompt_count)),
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
    return Panel(Text(response_text, style="white"), title=f"[bold cyan]{item.id}[/bold cyan] model response", border_style="bright_black", box=box.ROUNDED)


def render_result_row(item: PromptItem, score: Score) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("id",      no_wrap=True, style="bold cyan")
    table.add_column("status",  no_wrap=True)
    table.add_column("score",   no_wrap=True, justify="right")
    table.add_column("bar",     no_wrap=True)
    table.add_column("verdict", overflow="fold")
    color  = score_color(score.total)
    status = Text("×" if score.total == 0 else "✓", style=f"bold {color}")
    table.add_row(item.id, status, Text(f"{score.total}/100", style=f"bold {color}"), render_score_bar(score.total, width=24), Text(score.verdict or score_label(score.total), style="white"))
    return table


def render_score_bar(score: int, *, width: int = 24) -> Text:
    ratio  = min(1.0, max(0.0, score / 100))
    filled = int(round(width * ratio))
    return render_block_bar(filled=filled, width=width, complete_style=score_color(score), remaining_style="grey23")


def render_block_bar(*, filled: int, width: int, complete_style: str, remaining_style: str) -> Text:
    filled = max(0, min(width, filled))
    bar = Text()
    bar.append("[", style="bright_black")
    bar.append("█" * filled, style=complete_style)
    bar.append("░" * (width - filled), style=remaining_style)
    bar.append("]", style="bright_black")
    return bar


def render_multi_comparison(reports: list[dict[str, Any]]) -> Group:
    sorted_reports = sorted(reports, key=lambda r: r["summary"]["score_final"], reverse=True)
    has_runs = any(int(r["summary"].get("runs_count", 1)) > 1 for r in sorted_reports)

    tbl = Table(
        box=box.DOUBLE, border_style="bright_cyan",
        show_header=True, header_style="bold cyan",
        padding=(0, 1), expand=False,
    )
    tbl.add_column("#",       style="bold",  no_wrap=True, justify="center", width=4)
    tbl.add_column("MODEL",   style="white", no_wrap=True)
    if has_runs:
        tbl.add_column("RUNS", justify="right", no_wrap=True, width=5)
    tbl.add_column("SCORE",   justify="right", no_wrap=True, width=10)
    if has_runs:
        tbl.add_column("± σ",  justify="right", no_wrap=True, width=6)
    tbl.add_column("FORMAT",  justify="right", no_wrap=True, width=8)
    tbl.add_column("DENSITY", justify="right", no_wrap=True, width=9)
    tbl.add_column("TONE",    justify="right", no_wrap=True, width=7)
    tbl.add_column("",        no_wrap=True, width=26)

    rank_styles = ["bold yellow", "bold white", "bold cyan"]
    medals      = ["🥇", "🥈", "🥉"]

    for i, report in enumerate(sorted_reports):
        s     = report["summary"]
        final = int(s["score_final"])
        color = score_color(final)
        rstyle = rank_styles[i] if i < 3 else "dim"
        medal  = medals[i] if i < 3 else f"#{i + 1}"
        model_label = report.get("requested_model", report.get("model", "?"))
        runs_n  = int(s.get("runs_count", 1))
        std_val = s.get("stddev", {}).get("score_final")

        row = [
            Text(medal,  style=rstyle),
            Text(model_label, style="bold white" if i == 0 else "white"),
        ]
        if has_runs:
            row.append(Text(f"×{runs_n}" if runs_n > 1 else "—", style="bright_black"))
        row.append(Text(f"{final}%", style=f"bold {color}"))
        if has_runs:
            row.append(Text(f"±{std_val:.1f}" if std_val is not None and runs_n > 1 else "—", style="bright_black"))
        row.extend([
            Text(f"{s['format']}/33",  style="cyan"),
            Text(f"{s['densite']}/33", style="cyan"),
            Text(f"{s['ton']}/34",     style="cyan"),
            render_score_bar(final, width=22),
        ])
        tbl.add_row(*row)

    title_line = Text("  MULTI-MODEL COMPARISON ", style="bold bright_cyan")
    title_line.append(f"— {len(reports)} models", style="bright_black")

    return Group(
        Text(""),
        Align.left(title_line),
        Text(""),
        tbl,
        Text(""),
    )


def render_aggregated_summary(report: dict[str, Any], output_path: Path) -> Group:
    """Per-model aggregate summary across N runs, with mean + stddev + per-run scores."""
    s        = report["summary"]
    runs_n   = int(s.get("runs_count", 1))
    final    = int(s["score_final"])
    color    = score_color(final)
    std      = s.get("stddev", {})
    per_run  = s.get("per_run", [])

    score_line = Text("  ")
    score_line.append("AGGREGATE SCORE  ", style="bold cyan")
    score_line.append(f"{final}%   ", style=f"bold {color}")
    score_line.append(score_indicator(final), style=f"bold {color}")
    score_line.append(f" {s['interpretation']}  ", style=f"bold {color}")
    score_line.append(f"(n={runs_n})", style="bright_black")

    bits: list[Any] = [
        Text(""),
        render_separator(),
        score_line,
        render_kv_line("Format",    f"{s['format']}/33   ±{std.get('format', 0)}"),
        render_kv_line("Density",   f"{s['densite']}/33   ±{std.get('densite', 0)}"),
        render_kv_line("Tone",      f"{s['ton']}/34   ±{std.get('ton', 0)}"),
        render_kv_line("Score σ",   f"±{std.get('score_final', 0)}"),
        render_separator(),
    ]

    if per_run:
        per_run_line = Text("  ")
        per_run_line.append("PER-RUN          ", style="bold cyan")
        for i, pr in enumerate(per_run):
            per_run_line.append(f"r{i + 1}=", style="bright_black")
            per_run_line.append(f"{pr['score_final']}%", style=score_color(int(pr["score_final"])))
            if i < len(per_run) - 1:
                per_run_line.append("  ", style="bright_black")
        bits.append(per_run_line)
        bits.append(render_separator())

    bits.append(render_kv_line("Aggregated saved", str(output_path)))
    return Group(*bits)


def render_final_summary(report: dict[str, Any], output_path: Path) -> Group:
    summary     = report["summary"]
    final_score = int(summary["score_final"])
    color       = score_color(final_score)
    score_line  = Text("  ")
    score_line.append("FINAL SCORE     ", style="bold cyan")
    score_line.append(f"{final_score:>3}%   ", style=f"bold {color}")
    score_line.append(score_indicator(final_score), style=f"bold {color}")
    score_line.append(f" {summary['interpretation']}", style=f"bold {color}")
    return Group(
        Text(""),
        render_separator(),
        score_line,
        render_kv_line("Format",      f"{summary['format']}/33"),
        render_kv_line("Density",     f"{summary['densite']}/33"),
        render_kv_line("Tone",        f"{summary['ton']}/34"),
        render_separator(),
        render_kv_line("Report saved", str(output_path)),
    )


def score_indicator(score: int) -> str:
    if score >= 80: return "🟢"
    if score >= 60: return "🟡"
    if score >= 40: return "🟠"
    return "🔴"


def score_color(score: int) -> str:
    if score >= 80: return "green"
    if score >= 60: return "yellow"
    if score >= 40: return "orange3"
    return "red"


def score_label(score: int) -> str:
    if score >= 80: return "Very human"
    if score >= 60: return "Good, with visible AI tics"
    if score >= 40: return "Clearly AI-styled"
    return "Sedated PowerPoint"


# ── Legacy print_help (kept for backwards compat with run_benchmark.py) ────

def print_help(console: Console) -> None:
    _print_run_help(console)


configure_terminal_encoding()
