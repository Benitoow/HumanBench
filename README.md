# HumanBench

HumanBench is a terminal-first benchmark for measuring how closely an LLM sounds like a calibrated human rather than a polished product brochure.

It is intentionally opinionated:

- no web UI
- no dashboard
- no Electron wrapper
- just terminal output, JSON reports, and a scoring framework you can audit

## What it does

- runs a model against a curated prompt set
- asks a judge model to score the answer on three axes:
	- format
	- density
	- tone
- stores a machine-readable JSON report in `results/`
- supports multiple model providers through adapters

## Requirements

- Python 3.10+
- One or more provider API keys
- A terminal that can display ANSI colors well enough to enjoy the vibe

## Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Create your local environment file

```bash
copy .env.example .env
```

On PowerShell, you can also use:

```powershell
Copy-Item .env.example .env
```

### 4) Add your API keys

Edit `.env` and provide the providers you plan to use.

Example:

```env
PROVIDER=auto
MODEL_TESTED=anthropic/claude-sonnet-4-6
OPENROUTER_API_KEY=sk-or-xxxx
DEEPSEEK_API_KEY=sk-xxxx
ANTHROPIC_API_KEY=sk-ant-xxxx
OPENAI_API_KEY=sk-xxxx
MISTRAL_API_KEY=xxxx
GOOGLE_API_KEY=xxxx
```

## Run a benchmark

```bash
python run_benchmark.py --model anthropic/claude-sonnet-4-6
```

Useful variations:

```bash
python run_benchmark.py --model openai/gpt-4.1 --verbose
python run_benchmark.py --model mistralai/mistral-large --prompts prompts.json --output results/mistral.json
python run_benchmark.py --model xiaomi/mimo-v2-flash
python run_benchmark.py --provider deepseek --model deepseek-v4-flash
python run_benchmark.py --provider anthropic --model claude-sonnet-4-5
python run_benchmark.py --provider google --model gemini-2.5-flash
```

By default, the JSON report is written to `results/`.

## Providers and routing

HumanBench supports both OpenRouter and native providers via adapters.

Supported backends:

- `openrouter`
- `deepseek`
- `anthropic`
- `openai`
- `mistral`
- `google`

The tested model provider is configured through `PROVIDER`.

The judge is intentionally hardcoded in `run_benchmark.py` to `deepseek/deepseek-v4-pro` through OpenRouter, routed to the official `deepseek` provider with fallbacks disabled and `xhigh` reasoning. Changing the judge from run to run would turn the benchmark into confetti with a CLI.

If you use DeepSeek directly, set `PROVIDER=deepseek` with `DEEPSEEK_API_KEY` and a native model such as `deepseek-v4-pro` or `deepseek-v4-flash`.

## Output

Each run produces a JSON summary containing:

- prompt metadata
- tested model and fixed judge configuration
- per-prompt scores
- final aggregate score
- report timestamp

The report file is safe to archive and easy to diff across runs.

## Project structure

```text
HumanBench/
├── adapters/           # provider-specific model adapters
├── judge/              # judge prompt and scoring logic support
├── prompts.json        # benchmark prompts
├── results/            # generated benchmark reports
├── run_benchmark.py    # main CLI entrypoint
├── SCORING_FRAMEWORK.md
├── README.md
├── requirements.txt
└── .env.example
```

## Repository safety

HumanBench is open source, but the repository should stay clean:

- never commit `.env`
- keep API keys out of source control
- keep generated reports in `results/` out of commits unless they are intentional artifacts
- use `.env.example` as the only shared configuration template
- review diffs before merging to make sure no secrets or personal data slipped in

## Notes

- There is no Phase 3 roadmap here.
- The leaderboard concept was intentionally dropped for now.
- The project currently focuses on a clean terminal benchmark and a transparent scoring system.

## License

Add the license that matches your open-source release before publishing the repository publicly.
