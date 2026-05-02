# HUMANBENCH

> *Does your model sound like a human or a press release?*

HumanBench is a terminal-first benchmark that measures how closely an LLM sounds like a calibrated human — not a polished product brochure.

Intentionally opinionated. No web UI. No dashboard. No Electron wrapper.  
Just terminal output, JSON reports, and a scoring framework you can audit.

---

## How it works

HumanBench runs a model against a curated prompt set, then asks a fixed judge to score each answer on three axes:

| Axis | What it measures |
|------|-----------------|
| **Format** | Does the structure feel natural or template-generated? |
| **Density** | Is the information weight appropriate, or bloated? |
| **Tone** | Does it read like a human wrote it, or like a feature spec? |

Results are stored as machine-readable JSON in `results/` and rendered on the live leaderboard.

---

## Requirements

- Python 3.10+
- One or more provider API keys
- A terminal that renders ANSI colors well enough to enjoy the vibe

---

## Quick start

**1 — Create and activate a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

**2 — Install the package**

```bash
pip install -e .
```

This installs all dependencies and registers the `humanbench` command globally in your environment.

**3 — Configure your API keys**

```bash
humanbench config
```

An interactive wizard will walk you through picking a provider, entering your API key, and selecting a model. The wizard runs automatically on first launch if `.env` is missing or empty — but you can re-run it any time to change your setup.

---

## Running a benchmark

```bash
humanbench run anthropic/claude-sonnet-4-6
```

Other examples:

```bash
humanbench run openai/gpt-4.1 --verbose
humanbench run mistralai/mistral-large --prompts prompts.json --output results/mistral.json
humanbench run --provider deepseek --model deepseek-v4-flash
humanbench run --provider google --model gemini-2.5-flash
```

The JSON report is written to `results/` by default.

### Multiple runs and multiple models

Single-run scores carry noise from model temperature and judge variance. To reduce that, average several runs:

```bash
humanbench run anthropic/claude-opus-4-7 --runs 5
```

Each run produces its own report; an aggregated report (`<model>_<date>_avgN.json`) is then written with the **mean score and standard deviation per axis**. Only the aggregate is pushed to the leaderboard.

Compare several models in one shot — `--runs` applies to each:

```bash
humanbench run claude-sonnet-4-6 gpt-4.1 mistral-large --runs 3
```

A side-by-side comparison table prints at the end, ranked by score with the standard deviation shown when `--runs > 1`.

After a successful run, HumanBench also syncs the public leaderboard files
(`results.json` and `site/results.json`), commits those two files, and pushes
the current branch to `origin`.

To run locally without publishing:

```bash
humanbench run anthropic/claude-sonnet-4-6 --no-push
```

You can also disable auto-push from `.env`:

```bash
HUMANBENCH_AUTO_PUSH=0
```

---

## Commands

| Command | Description |
|---------|-------------|
| `humanbench run [model]` | Run the benchmark on the specified model |
| `humanbench config` | Re-run the interactive setup wizard |
| `humanbench --help` | Top-level help |
| `humanbench run --help` | Options for the run command |

`--model` is also accepted as a flag for backwards compatibility:

```bash
humanbench run --model anthropic/claude-sonnet-4-6
```

---

## Providers

| Backend | Key in `.env` | Notes |
|---------|--------------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | Routes to any provider |
| `anthropic` | `ANTHROPIC_API_KEY` | Native Anthropic API |
| `openai` | `OPENAI_API_KEY` | Native OpenAI API |
| `mistral` | `MISTRAL_API_KEY` | Native Mistral API |
| `google` | `GOOGLE_API_KEY` | Native Gemini API |
| `deepseek` | `DEEPSEEK_API_KEY` | Native DeepSeek API |

---

## Output

Each run produces a JSON summary with:

- prompt metadata
- tested model and fixed judge configuration
- per-prompt scores (format, density, tone)
- final aggregate score
- report timestamp

The file is safe to archive and easy to diff across runs.

---

## Submit a Result

To get your result on the public leaderboard:

**1 — Run the benchmark**

```bash
humanbench run your-org/your-model
```

**2 — Submit via the site**

Click **[ SUBMIT A RUN ]** on the leaderboard — in the header or below the table. Fill in:

- **Model name** — the `org/model` string, e.g. `anthropic/claude-sonnet-4-6`
- **Provider** — the backend used, e.g. `openrouter`, `anthropic`
- **Result JSON** — the `.json` file from `results/`

You will see: `RUN RECEIVED // PENDING VALIDATION`

**3 — Validation**

Results are reviewed manually before appearing on the leaderboard. Scores are verified against the benchmark protocol.

---

## Project structure

```
HumanBench/
├── humanbench/          # installable package
│   ├── __init__.py
│   └── cli.py           # humanbench command entry point
├── adapters/            # provider-specific model adapters
├── judge/               # judge prompt and scoring support
├── site/                # leaderboard website
├── prompts.json         # benchmark prompt set
├── results/             # generated reports (gitignored)
├── run_benchmark.py     # legacy shim (delegates to humanbench.cli)
├── pyproject.toml
├── SCORING_FRAMEWORK.md
├── requirements.txt
└── .env.example
```

---

## Repository hygiene

- Never commit `.env` or any file containing API keys
- Keep generated reports in `results/` out of commits unless they are intentional artifacts
- Auto-publish commits only the public leaderboard files; raw `results/*.json` reports stay gitignored
- Use `.env.example` as the only shared config template
- Review diffs before merging — secrets slip in quietly

---

## License

Add the license that matches your open-source release before publishing publicly.
