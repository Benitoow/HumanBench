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

**2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**3 — Set up environment**

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env` and fill in the providers you plan to use:

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

---

## Running a benchmark

```bash
python run_benchmark.py --model anthropic/claude-sonnet-4-6
```

Other examples:

```bash
python run_benchmark.py --model openai/gpt-4.1 --verbose
python run_benchmark.py --model mistralai/mistral-large --prompts prompts.json --output results/mistral.json
python run_benchmark.py --model xiaomi/mimo-v2-flash
python run_benchmark.py --provider deepseek --model deepseek-v4-flash
python run_benchmark.py --provider anthropic --model claude-sonnet-4-5
python run_benchmark.py --provider google --model gemini-2.5-flash
```

The JSON report is written to `results/` by default.

---

## Providers

| Backend | Key in `.env` | Notes |
|---------|--------------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | Routes to any provider |
| `deepseek` | `DEEPSEEK_API_KEY` | Native DeepSeek API |
| `anthropic` | `ANTHROPIC_API_KEY` | Native Anthropic API |
| `openai` | `OPENAI_API_KEY` | Native OpenAI API |
| `mistral` | `MISTRAL_API_KEY` | Native Mistral API |
| `google` | `GOOGLE_API_KEY` | Native Gemini API |

Set the active backend with `PROVIDER=` in `.env`.

> **Judge** — hardcoded to `deepseek/deepseek-v4-pro` via OpenRouter, fallbacks disabled, `xhigh` reasoning. This is intentional. Swapping judges between runs makes scores incomparable.

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
python run_benchmark.py --model your-org/your-model
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
├── adapters/            # provider-specific model adapters
├── judge/               # judge prompt and scoring support
├── site/                # leaderboard website
├── prompts.json         # benchmark prompt set
├── results/             # generated reports (gitignored)
├── run_benchmark.py     # CLI entrypoint
├── SCORING_FRAMEWORK.md
├── requirements.txt
└── .env.example
```

---

## Repository hygiene

- Never commit `.env` or any file containing API keys
- Keep generated reports in `results/` out of commits unless they are intentional artifacts
- Use `.env.example` as the only shared config template
- Review diffs before merging — secrets slip in quietly

---

## License

Add the license that matches your open-source release before publishing publicly.
