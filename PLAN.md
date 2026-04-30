# HumanBench - Plan de Projet

> Benchmark open source pour évaluer si une IA répond comme un humain calibré, pas comme un PowerPoint.
> Interface : terminal uniquement. Pas de web, pas de dashboard. Du beau ASCII et des couleurs.

---

## Concept

Un LLM répond à une série de prompts typés (situations réelles de conversation).
Un LLM juge évalue chaque réponse sur 3 dimensions.
Le score final = pourcentage de "humanité". 100% = humain de référence.

---

## Architecture Globale

```
[prompts.json] --> [modèle testé via API] --> [réponses] --> [juge LLM] --> [score affiché en terminal]
```

---

## Stack Technique

- **Python** uniquement
- **Rich** pour le terminal (couleurs, tableaux, progress bar, gros texte, panels)
- **OpenRouter** en Phase 1 (une clé, tous les modèles)
- **APIs natives** en Phase 2
- **Fichiers JSON** pour stocker prompts et résultats
- Zéro frontend, zéro web, zéro Electron

---

## Ce que voit l'utilisateur dans le terminal

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ██╗  ██╗██╗   ██╗███╗   ███╗ █████╗ ███╗   ██╗        ║
║   ██║  ██║██║   ██║████╗ ████║██╔══██╗████╗  ██║        ║
║   ███████║██║   ██║██╔████╔██║███████║██╔██╗ ██║        ║
║   ██╔══██║██║   ██║██║╚██╔╝██║██╔══██║██║╚██╗██║        ║
║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║██║ ╚████║        ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝        ║
║                                                          ║
║          BENCHMARK   •   HUMANNESS   •   v0.1            ║
╚══════════════════════════════════════════════════════════╝

  Modèle testé   : claude/claude-sonnet-4-6
  Modèle juge    : deepseek/deepseek-r1
  Prompts        : 25
  ──────────────────────────────────────────────────────

  [████████████████████░░░░] 20/25 prompts   80%

  EMO_001    ✓   87/100   [██████████████████░░]
  EMO_002    ✓   91/100   [███████████████████░]
  SIMPLE_001 ✓   73/100   [██████████████░░░░░░]
  ...

  ──────────────────────────────────────────────────────
  SCORE FINAL         88%   🟢 Très humain
  Format              29/33
  Densité             31/33
  Ton                 28/34
  ──────────────────────────────────────────────────────
  Rapport sauvegardé : results/claude-sonnet_2026-04-30.json
```

---

## Phase 1 - MVP Terminal

### Commande de lancement
```bash
python run_benchmark.py --model anthropic/claude-sonnet-4-6
```

### Options CLI
```
--model     modèle à tester (format OpenRouter)
--judge     override le juge par défaut (optionnel)
--prompts   chemin vers un fichier prompts custom (optionnel)
--verbose   affiche les réponses complètes en temps réel
--output    chemin de sortie du rapport JSON
```

### Librairies Python à installer
```
rich          # tout le visuel terminal
httpx         # appels API
python-dotenv # lecture du .env
```

### Configuration (.env)
```
OPENROUTER_API_KEY=sk-or-xxxx
MODEL_JUDGE=deepseek/deepseek-r1
```

---

## Phase 2 - Support Multi-API

Adapter pattern pour brancher les APIs natives sans passer par OpenRouter.

```python
class ModelAdapter:
    def generate(self, prompt: str) -> str: ...

class OpenRouterAdapter(ModelAdapter): ...
class AnthropicAdapter(ModelAdapter): ...
class OpenAIAdapter(ModelAdapter): ...
class MistralAdapter(ModelAdapter): ...
class GoogleAdapter(ModelAdapter): ...
```

Config étendue dans .env :
```
PROVIDER=anthropic
API_KEY=sk-ant-xxxx
MODEL_TESTED=claude-opus-4-6
```

Le script détecte le provider automatiquement et instancie le bon adapter.

---

## Phase 3 - Leaderboard Terminal

Pas de web. Un fichier leaderboard.json local qui agrège tous les résultats passés.

```bash
python leaderboard.py
```

Affiche dans le terminal :

```
╔══════════════════════════════════════════════════╗
║              HUMANBENCH LEADERBOARD              ║
╠══════════════════════════════════════════════════╣
║  #1   claude-opus-4-6          94%   🟢          ║
║  #2   gpt-5.3                  81%   🟢          ║
║  #3   gemini-2.5-flash         74%   🟡          ║
║  #4   mistral-small            61%   🟡          ║
║  #5   gpt-3.5-turbo            38%   🔴          ║
╚══════════════════════════════════════════════════╝
```

Chacun peut soumettre ses résultats via un fichier JSON standardisé.
La communauté valide manuellement avant merge sur le repo GitHub.

---

## Structure des Fichiers

```
humanbench/
├── PLAN.md
├── SCORING_FRAMEWORK.md
├── README.md
├── .env.example
├── prompts.json
├── run_benchmark.py        # script principal
├── leaderboard.py          # affiche le leaderboard local
├── adapters/
│   ├── __init__.py
│   ├── base.py
│   ├── openrouter.py
│   ├── anthropic.py
│   ├── openai.py
│   ├── mistral.py
│   └── google.py
├── judge/
│   └── prompt.txt
└── results/
    └── [model]_[date].json
```

---