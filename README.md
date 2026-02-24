# LLM Chess Arena

Run two LLMs against each other in chess using OpenRouter and python-chess, with a simple CLI spectator view.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your_key"
```

Optional headers for OpenRouter analytics:

```bash
export OPENROUTER_APP_URL="https://your-app.example"
export OPENROUTER_APP_TITLE="LLM Chess Arena"
```

Install Stockfish and ensure it is on your PATH, or drop a `stockfish/` folder with the binary in the repo.
You can also set:

```bash
export STOCKFISH_PATH="/path/to/stockfish"
export STOCKFISH_DEPTH=15
```

## Run a Match

```bash
llm-chess \
  --model-a openai/gpt-4o-mini \
  --model-b anthropic/claude-3.5-sonnet \
  --temperature-a 0.2 \
  --temperature-b 0.2 \
  --delay 0.4
```

Headless series (no rendering, no delays, runs 10 games in parallel):

```bash
llm-chess --headless --games 10
```

Headless mode uses a model pool from `.llm_chess/models.json` instead of `--model-a/--model-b`. Example:

```json
{
  "models": [
    {
      "model": "x-ai/grok-4.1-fast",
      "temperature": 0.2,
      "reasoning_effort": "high",
      "reasoning_include": false
    },
    {
      "model": "google/gemini-3-flash-preview",
      "temperature": 0.2,
      "reasoning_max_tokens": 2000
    }
  ]
}
```

Model fields: `model` (required), `temperature` (optional), `reasoning_effort` or `reasoning_max_tokens` (optional),
`reasoning_include` (optional), `reasoning_enabled` (optional, default true).
In headless mode, the runner picks the two models with the fewest games in the leaderboard and randomizes colors.

Without installing, you can run:

```bash
PYTHONPATH=src python -m llm_chess --model-a openai/gpt-4o-mini --model-b anthropic/claude-3.5-sonnet
```

## CLI Options

- `--model-a`, `--model-b`: OpenRouter model IDs
- `--temperature-a`, `--temperature-b`: Sampling temperature per model
- `--max-plies`: Stop after N plies
- `--delay`: Delay between moves
- `--headless`: Run X games without rendering and only report scores
- `--games`: Number of games to play in headless mode
- `--reasoning-effort-a`, `--reasoning-effort-b`: Reasoning effort per model
- `--reasoning-max-tokens-a`, `--reasoning-max-tokens-b`: Max reasoning tokens per model
- `--no-reasoning`: Disable reasoning (default is enabled)
- `--reasoning-include`: Include reasoning tokens in responses (default is excluded)
- `--no-clear`: Keep all output in the terminal
- `--request-timeout`: HTTP timeout in seconds
- `--seed`: Random seed for fallback move selection

## Notes

- The CLI prints the board, recent moves, and the last raw model reply.
- If a model returns an illegal move, the runner falls back to the first legal move.
- Elo ratings are tracked per model in `.llm_chess/leaderboard.json` and printed after each run.
- Reasoning is enabled by default; disable with `--no-reasoning` or `reasoning_enabled: false` in models.json.
- Avg cost is tracked per model as cost-per-move, averaged across games.
- Accuracy is tracked with Stockfish (ACPL-based). Set `STOCKFISH_PATH` and optional `STOCKFISH_DEPTH`.
