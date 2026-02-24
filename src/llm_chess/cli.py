import argparse
import base64
import io
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
from stockfish import Stockfish
from openai import OpenAI


# Stockfish players start at their target ELO rating
# Note: Stockfish library minimum ELO is 1320
STOCKFISH_STARTING_ELOS = {
    "stockfish/elo-1350": 1350.0,
    "stockfish/elo-1500": 1500.0,
    "stockfish/elo-1750": 1750.0,
    "stockfish/elo-2000": 2000.0,
    "stockfish/elo-2250": 2250.0,
    "stockfish/elo-2500": 2500.0,
    "stockfish/elo-2750": 2750.0,
    "stockfish/elo-3000": 3000.0,
}


def get_starting_elo(model_id: str) -> float:
    """Get starting ELO for a model. Returns 1200 for LLMs, specific ELO for Stockfish."""
    return STOCKFISH_STARTING_ELOS.get(model_id, 1200.0)


@dataclass
class PlayerConfig:
    model: str
    temperature: float
    name: str
    reasoning: Optional[Dict[str, object]]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    temperature: float
    reasoning: Optional[Dict[str, object]]

class OpenRouterChessClient:
    def __init__(self, api_key: str, base_url: str, app_url: Optional[str], app_title: Optional[str]):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._headers = {}
        self._image_support: Dict[str, bool] = {}
        self._structured_support: Dict[str, bool] = {}
        self._log_images = False
        self._save_image_dir = None
        self._logger = logging.getLogger("llm_chess.openrouter")
        if self._log_images and not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if app_url:
            self._headers["HTTP-Referer"] = app_url
        if app_title:
            self._headers["X-Title"] = app_title

    @staticmethod
    def _extract_cost(response: object) -> Optional[float]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if isinstance(usage, dict):
            for key in ("total_cost", "cost"):
                if key in usage and usage[key] is not None:
                    return float(usage[key])
        for key in ("total_cost", "cost"):
            value = getattr(usage, key, None)
            if value is not None:
                return float(value)
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("total_cost", "cost"):
                if key in extra and extra[key] is not None:
                    return float(extra[key])
        return None

    def ask_move(
        self,
        model: str,
        prompt: str,
        temperature: float,
        timeout: Optional[float],
        reasoning: Optional[Dict[str, object]],
        board_image_url: Optional[str] = None,
    ) -> Tuple[str, Optional[float], float, Optional[Dict[str, object]]]:
        def build_messages(include_image: bool) -> List[Dict[str, object]]:
            system = {
                "role": "system",
                "content": (
                    "You are a chess engine. Reply ONLY as a JSON object with keys: "
                    "`move` (UCI string) and optional `banter` (short, witty heckle). "
                    "If you need to resign or offer a draw, call the appropriate tool. "
                    "If a draw is offered, respond with accept_draw or decline_draw. "
                    "No extra text."
                ),
            }
            if include_image and board_image_url:
                return [
                    system,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": board_image_url}},
                        ],
                    },
                ]
            return [system, {"role": "user", "content": prompt}]

        extra_body = {"reasoning": reasoning} if reasoning else None
        include_image = bool(board_image_url) and self._image_support.get(model, True)
        include_structured = self._structured_support.get(model, True)
        if include_image and self._log_images:
            content_type = "unknown"
            if board_image_url and board_image_url.startswith("data:"):
                content_type = board_image_url.split(";", 1)[0].replace("data:", "")
            self._logger.info(
                "Sending board image to model %s (%s, data URL length %s).",
                model,
                content_type,
                len(board_image_url or ""),
            )
        elif board_image_url and self._log_images:
            self._logger.info("Skipping board image for model %s (image disabled or unsupported).", model)
        if include_image and self._save_image_dir:
            self._save_board_image(model, board_image_url)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "chess_move",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "move": {"type": "string", "description": "Move in UCI format."},
                        "banter": {
                            "type": "string",
                            "description": "Optional witty heckle about the position or opponent.",
                        },
                    },
                    "required": ["move"],
                    "additionalProperties": False,
                },
            },
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "resign",
                    "description": "Resign the game immediately if the position is hopeless.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Short reason for resigning.",
                            }
                        },
                        "required": ["reason"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "offer_draw",
                    "description": "Offer a draw and explain why the position is drawn.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Short draw-offer message explaining why it's a draw.",
                            }
                        },
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "accept_draw",
                    "description": "Accept the opponent's draw offer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Optional acceptance message.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "decline_draw",
                    "description": "Decline the opponent's draw offer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Optional decline message.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
        ]

        attempts: List[Tuple[bool, bool]] = []
        if include_structured:
            attempts.append((include_image, True))
            if include_image:
                attempts.append((False, True))
        attempts.append((include_image, False))
        if include_image:
            attempts.append((False, False))
        if not attempts:
            attempts = [(False, False)]

        response = None
        used_image = None
        used_structured = None
        last_error: Optional[Exception] = None
        start = time.perf_counter()
        for use_image, use_structured in attempts:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=build_messages(use_image),
                    temperature=temperature,
                    response_format=response_format if use_structured else None,
                    tools=tools,
                    tool_choice="auto",
                    extra_headers=self._headers or None,
                    extra_body=extra_body,
                    timeout=timeout,
                )
                used_image = use_image
                used_structured = use_structured
                break
            except Exception as exc:
                last_error = exc
                continue
        if response is None:
            raise last_error or RuntimeError("OpenRouter request failed.")
        elapsed = time.perf_counter() - start
        if include_image and used_image is False:
            self._image_support[model] = False
            if self._log_images:
                self._logger.info("Model %s rejected image input; using text-only.", model)
        if include_structured and used_structured is False:
            self._structured_support[model] = False
            if self._log_images:
                self._logger.info("Model %s rejected structured outputs; using plain JSON prompt.", model)
        cost = self._extract_cost(response)
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            if self._log_images:
                self._logger.warning("Model %s returned no choices; falling back to empty reply.", model)
            return "", cost, elapsed, None
        message = choices[0].message
        if message is None:
            if self._log_images:
                self._logger.warning("Model %s returned empty message; falling back to empty reply.", model)
            return "", cost, elapsed, None
        tool_action = None
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            call = tool_calls[0]
            function = getattr(call, "function", None)
            name = getattr(function, "name", None) or getattr(call, "name", None)
            arguments = getattr(function, "arguments", None) or getattr(call, "arguments", None) or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"message": arguments}
            if isinstance(name, str):
                tool_action = {"name": name, "arguments": arguments}
                self._logger.info("Model %s called tool %s.", model, name)
        return message.content or "", cost, elapsed, tool_action

    def _save_board_image(self, model: str, board_image_url: Optional[str]) -> None:
        if not self._save_image_dir or not board_image_url:
            return
        if not board_image_url.startswith("data:"):
            return
        header, _, payload = board_image_url.partition(",")
        if not header or not payload:
            return
        content_type = header.split(";", 1)[0].replace("data:", "")
        ext_map = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
            "image/gif": "gif",
            "image/svg+xml": "svg",
        }
        ext = ext_map.get(content_type, "bin")
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
        timestamp = int(time.time() * 1000)
        directory = Path(self._save_image_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / f"{safe_model}_{timestamp}.{ext}"
        try:
            data = base64.b64decode(payload)
        except Exception:
            return
        filename.write_bytes(data)
        if self._log_images:
            self._logger.info("Saved board image to %s", filename)

def leaderboard_path() -> Path:
    return Path(".llm_chess") / "leaderboard.json"

def load_leaderboard(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data.get("players", {})

def save_leaderboard(path: Path, players: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"players": players}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def ensure_player(
    players: Dict[str, Dict[str, float]],
    model_id: str,
    starting_elo: Optional[float] = None,
) -> Dict[str, float]:
    # Use provided starting_elo, or get from Stockfish lookup, or default to 1200
    default_elo = starting_elo if starting_elo is not None else get_starting_elo(model_id)
    defaults = {
        "elo": default_elo,
        "games": 0.0,
        "wins": 0.0,
        "losses": 0.0,
        "draws": 0.0,
        "cost_avg": 0.0,
        "cost_games": 0.0,
        "accuracy_avg": 0.0,
        "accuracy_games": 0.0,
        "time_avg": 0.0,
        "time_games": 0.0,
    }
    stats = players.get(model_id)
    if stats is None:
        stats = {}
        players[model_id] = stats
    for key, value in defaults.items():
        stats.setdefault(key, value)
    return stats

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

def update_elo(
    players: Dict[str, Dict[str, float]],
    model_a: str,
    model_b: str,
    score_a: float,
    k_factor: float = 32.0,
) -> None:
    if model_a == model_b:
        stats = ensure_player(players, model_a)
        stats["games"] += 1.0
        stats["draws"] += 1.0
        return
    stats_a = ensure_player(players, model_a)
    stats_b = ensure_player(players, model_b)
    expected_a = expected_score(stats_a["elo"], stats_b["elo"])
    expected_b = 1.0 - expected_a
    stats_a["elo"] += k_factor * (score_a - expected_a)
    stats_b["elo"] += k_factor * ((1.0 - score_a) - expected_b)
    stats_a["games"] += 1.0
    stats_b["games"] += 1.0
    if score_a == 1.0:
        stats_a["wins"] += 1.0
        stats_b["losses"] += 1.0
    elif score_a == 0.0:
        stats_a["losses"] += 1.0
        stats_b["wins"] += 1.0
    else:
        stats_a["draws"] += 1.0
        stats_b["draws"] += 1.0

def apply_result(
    players: Dict[str, Dict[str, float]],
    model_white: str,
    model_black: str,
    result: str,
) -> None:
    if result == "1-0":
        score_white = 1.0
    elif result == "0-1":
        score_white = 0.0
    else:
        score_white = 0.5
    print(f"[DEBUG apply_result] result={result}, score_white={score_white}")
    print(f"[DEBUG apply_result] white={model_white}, black={model_black}")
    update_elo(players, model_white, model_black, score_white)

def update_cost_average(
    players: Dict[str, Dict[str, float]],
    model_id: str,
    avg_cost: float,
) -> None:
    stats = ensure_player(players, model_id)
    games = float(stats.get("cost_games", 0.0))
    previous = float(stats.get("cost_avg", 0.0))
    stats["cost_avg"] = (previous * games + avg_cost) / (games + 1.0)
    stats["cost_games"] = games + 1.0

def update_accuracy_average(
    players: Dict[str, Dict[str, float]],
    model_id: str,
    avg_accuracy: float,
) -> None:
    stats = ensure_player(players, model_id)
    games = float(stats.get("accuracy_games", 0.0))
    previous = float(stats.get("accuracy_avg", 0.0))
    stats["accuracy_avg"] = (previous * games + avg_accuracy) / (games + 1.0)
    stats["accuracy_games"] = games + 1.0


def update_time_average(
    players: Dict[str, Dict[str, float]],
    model_id: str,
    avg_time: float,
) -> None:
    stats = ensure_player(players, model_id)
    games = float(stats.get("time_games", 0.0))
    previous = float(stats.get("time_avg", 0.0))
    stats["time_avg"] = (previous * games + avg_time) / (games + 1.0)
    stats["time_games"] = games + 1.0

def print_leaderboard() -> None:
    path = leaderboard_path()
    players = load_leaderboard(path)
    if not players:
        print("Leaderboard: no games recorded yet.")
        return
    sorted_rows = sorted(players.items(), key=lambda item: item[1]["elo"], reverse=True)
    print("Leaderboard:")
    for idx, (model_id, stats) in enumerate(sorted_rows, start=1):
        print(
            f"{idx:>2}. {model_id} | Elo {stats['elo']:.1f} | "
            f"W {int(stats['wins'])} D {int(stats['draws'])} L {int(stats['losses'])}"
        )
    print(f"Saved to: {path}")

def models_path() -> Path:
    return Path(".llm_chess") / "models.json"

def build_reasoning_from_spec(spec: Dict[str, object]) -> Optional[Dict[str, object]]:
    effort = spec.get("reasoning_effort")
    max_tokens = spec.get("reasoning_max_tokens")
    include = bool(spec.get("reasoning_include", False))
    enabled = spec.get("reasoning_enabled", True)
    if effort and max_tokens:
        raise SystemExit("Model spec must not set both reasoning_effort and reasoning_max_tokens.")
    if enabled is False:
        if effort or max_tokens or include:
            raise SystemExit("Model spec disables reasoning but includes reasoning settings.")
        return None
    if not effort and not max_tokens:
        return {"enabled": True, "exclude": not include}
    config: Dict[str, object] = {"enabled": True, "exclude": not include}
    if effort:
        config["effort"] = effort
    if max_tokens:
        config["max_tokens"] = max_tokens
    return config

def load_model_pool(path: Path) -> List[ModelSpec]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Create a models list for headless mode.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict) or "models" not in data:
        raise SystemExit(f"{path} must contain a top-level 'models' list.")
    raw_models = data["models"]
    if not isinstance(raw_models, list) or len(raw_models) < 2:
        raise SystemExit("Model pool must contain at least two models.")
    models: List[ModelSpec] = []
    for item in raw_models:
        if not isinstance(item, dict):
            raise SystemExit("Each model entry must be an object.")
        model_id = item.get("model") or item.get("id")
        if not model_id:
            raise SystemExit("Each model entry must include a 'model' string.")
        temperature = float(item.get("temperature", 0.4))
        reasoning = build_reasoning_from_spec(item)
        models.append(ModelSpec(model=model_id, temperature=temperature, reasoning=reasoning))
    return models

def model_games(
    players: Dict[str, Dict[str, float]],
    model_id: str,
    scheduled_games: Optional[Dict[str, int]] = None,
) -> float:
    base = float(players.get(model_id, {}).get("games", 0.0))
    if scheduled_games:
        base += float(scheduled_games.get(model_id, 0))
    return base

def model_elo(players: Dict[str, Dict[str, float]], model_id: str) -> float:
    return float(players.get(model_id, {}).get("elo", 1200.0))

def stockfish_path() -> str:
    env_path = "stockfish/stockfish-macos-m1-apple-silicon"
    return env_path

def stockfish_depth() -> int:
    try:
        return int(os.getenv("STOCKFISH_DEPTH", "15"))
    except ValueError:
        return 15


def create_stockfish() -> Stockfish:
    path = stockfish_path()
    try:
        engine = Stockfish(path=path)
    except Exception as exc:
        raise SystemExit(
            f"Stockfish not found at '{path}'. Set STOCKFISH_PATH to the engine binary."
        ) from exc
    engine.set_depth(stockfish_depth())
    return engine


def eval_to_cp(evaluation: Dict[str, int]) -> int:
    if evaluation.get("type") == "cp":
        return int(evaluation.get("value", 0))
    if evaluation.get("type") == "mate":
        return 2000 if evaluation.get("value", 0) > 0 else -2000
    return 0


def eval_to_cp_for_turn(evaluation: Dict[str, int], turn: bool) -> int:
    cp = eval_to_cp(evaluation)
    return -cp if turn == chess.BLACK else cp


def eval_summary_for_turn(evaluation: Dict[str, int], turn: bool) -> Dict[str, Optional[float]]:
    eval_type = evaluation.get("type", "cp") if isinstance(evaluation, dict) else "cp"
    value = evaluation.get("value") if isinstance(evaluation, dict) else 0
    try:
        cp = eval_to_cp(evaluation)
    except (TypeError, ValueError):
        cp = 0
    if turn == chess.BLACK:
        cp = -cp
        if isinstance(value, (int, float)):
            value = -value
    return {"type": eval_type, "value": value, "cp": cp}


def format_eval_text(summary: Optional[Dict[str, Optional[float]]]) -> str:
    if not summary:
        return "n/a"
    eval_type = summary.get("type")
    value = summary.get("value")
    cp = summary.get("cp")
    if eval_type == "mate":
        if isinstance(value, (int, float)):
            return f"M{int(value):+d}"
        return "M?"
    if isinstance(cp, (int, float)):
        return f"{float(cp) / 100.0:+.2f}"
    if isinstance(value, (int, float)):
        return f"{float(value) / 100.0:+.2f}"
    return "n/a"


def tiny_png_data_url() -> str:
    encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5f6n8AAAAASUVORK5CYII="
    return f"data:image/png;base64,{encoded}"


def accuracy_from_losses(losses: List[int]) -> Optional[float]:
    if not losses:
        return None
    per_move = []
    for loss in losses:
        score = 100.0 - (loss / 1.5)
        per_move.append(max(0.0, min(100.0, score)))
    return sum(per_move) / float(len(per_move))


def compute_accuracy_report(
    moves_uci: List[str],
    white_model: str,
    black_model: str,
) -> Dict[str, Optional[float]]:
    if not moves_uci:
        return {white_model: None, black_model: None}
    engine = create_stockfish()
    engine.set_fen_position(chess.STARTING_FEN)
    white_losses: List[int] = []
    black_losses: List[int] = []
    seen_white = False
    seen_black = False

    for idx, move in enumerate(moves_uci):
        is_white = idx % 2 == 0
        before = eval_to_cp_for_turn(engine.get_evaluation(), chess.WHITE if is_white else chess.BLACK)
        engine.make_moves_from_current_position([move])
        after = eval_to_cp_for_turn(engine.get_evaluation(), chess.BLACK if is_white else chess.WHITE)
        if is_white and not seen_white:
            loss = 0
            seen_white = True
        elif (not is_white) and not seen_black:
            loss = 0
            seen_black = True
        else:
            loss = before - after if is_white else after - before
            loss = max(0, loss)
        if is_white:
            white_losses.append(loss)
        else:
            black_losses.append(loss)

    return {
        white_model: accuracy_from_losses(white_losses),
        black_model: accuracy_from_losses(black_losses),
    }


def choose_model_pair(
    models: List[ModelSpec],
    players: Dict[str, Dict[str, float]],
    scheduled_games: Optional[Dict[str, int]] = None,
) -> Tuple[ModelSpec, ModelSpec]:
    min_games = min(model_games(players, model.model, scheduled_games) for model in models)
    first_candidates = [
        model for model in models if model_games(players, model.model, scheduled_games) == min_games
    ]
    first = random.choice(first_candidates)
    remaining = [model for model in models if model.model != first.model]
    first_elo = model_elo(players, first.model)
    windows = [100, 200, 400, 800, 1600]
    second_candidates: List[ModelSpec] = []
    for window in windows:
        eligible = [
            model
            for model in remaining
            if abs(model_elo(players, model.model) - first_elo) <= window
        ]
        if not eligible:
            continue
        min_games_remaining = min(
            model_games(players, model.model, scheduled_games) for model in eligible
        )
        second_candidates = [
            model
            for model in eligible
            if model_games(players, model.model, scheduled_games) == min_games_remaining
        ]
        break
    if not second_candidates:
        min_games_remaining = min(
            model_games(players, model.model, scheduled_games) for model in remaining
        )
        second_candidates = [
            model
            for model in remaining
            if model_games(players, model.model, scheduled_games) == min_games_remaining
        ]
    min_delta = min(abs(model_elo(players, model.model) - first_elo) for model in second_candidates)
    closest = [
        model for model in second_candidates if abs(model_elo(players, model.model) - first_elo) == min_delta
    ]
    second = random.choice(closest)
    return first, second


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM vs LLM chess match via OpenRouter.")
    parser.add_argument("--model-a", default="openai/gpt-5.2", help="White model name.")
    parser.add_argument("--model-b", default="google/gemini-3-flash-preview", help="Black model name.")
    parser.add_argument("--temperature-a", type=float, default=0.5, help="White model temperature.")
    parser.add_argument("--temperature-b", type=float, default=0.5, help="Black model temperature.")
    parser.add_argument("--max-plies", type=int, default=100, help="Maximum plies before stopping.")
    parser.add_argument("--delay", type=float, default=0, help="Delay between moves in seconds.")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play in headless mode.")
    parser.add_argument("--headless", action="store_true", help="Run without rendering and report scores only.")
    parser.add_argument(
        "--reasoning-effort-a",
        choices=["xhigh", "high", "medium", "low", "minimal", "none"],
        default=None,
        help="Reasoning effort level for model A.",
    )
    parser.add_argument(
        "--reasoning-effort-b",
        choices=["xhigh", "high", "medium", "low", "minimal", "none"],
        default=None,
        help="Reasoning effort level for model B.",
    )
    parser.add_argument(
        "--reasoning-max-tokens-a",
        type=int,
        default=None,
        help="Max reasoning tokens for model A.",
    )
    parser.add_argument(
        "--reasoning-max-tokens-b",
        type=int,
        default=None,
        help="Max reasoning tokens for model B.",
    )
    parser.add_argument(
        "--reasoning-include",
        action="store_true",
        help="Include reasoning tokens in the response (overrides default exclusion).",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable reasoning entirely.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for fallbacks.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the screen between moves.")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="OpenRouter request timeout.")
    parser.add_argument("--app-url", default=os.getenv("OPENROUTER_APP_URL"), help="OpenRouter app URL header.")
    parser.add_argument("--app-title", default=os.getenv("OPENROUTER_APP_TITLE"), help="OpenRouter app title header.")
    return parser.parse_args(argv)


def build_reasoning_config(
    disabled: bool,
    effort: Optional[str],
    max_tokens: Optional[int],
    include: bool,
) -> Optional[Dict[str, object]]:
    if effort and max_tokens:
        raise SystemExit("Use only one of --reasoning-effort-* or --reasoning-max-tokens-*.")
    if disabled:
        if effort or max_tokens or include:
            raise SystemExit("Disable reasoning without additional reasoning settings.")
        return None
    config: Dict[str, object] = {"enabled": True}
    if effort:
        config["effort"] = effort
    if max_tokens:
        config["max_tokens"] = max_tokens
    config["exclude"] = not include
    return config


def board_to_ascii(board: chess.Board) -> str:
    rows = []
    for rank in range(8, 0, -1):
        row = [str(rank)]
        for file_idx in range(1, 9):
            square = chess.square(file_idx - 1, rank - 1)
            piece = board.piece_at(square)
            row.append(piece.symbol() if piece else ".")
        rows.append(" ".join(row))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def board_to_table(board: chess.Board) -> str:
    files = "a b c d e f g h"
    rows = [f"   {files}"]
    for rank in range(8, 0, -1):
        row: List[str] = []
        for file_idx in range(1, 9):
            square = chess.square(file_idx - 1, rank - 1)
            piece = board.piece_at(square)
            if not piece:
                row.append("..")
            else:
                color = "w" if piece.color == chess.WHITE else "b"
                row.append(f"{color}{piece.symbol().upper()}")
        rows.append(f"{rank}  " + " ".join(row))
    rows.append(f"   {files}")
    return "\n".join(rows)


def format_history(board: chess.Board, max_plies: int = 20) -> str:
    temp_board = chess.Board()
    sans: List[str] = []
    for move in board.move_stack:
        sans.append(temp_board.san(move))
        temp_board.push(move)
    if not sans:
        return "(none)"
    start = max(0, len(sans) - max_plies)
    sans = sans[start:]
    move_no = 1 + (start // 2)
    parts: List[str] = []
    idx = 0
    if start % 2 == 1:
        parts.append(f"{move_no}... {sans[0]}")
        idx = 1
        move_no += 1
    for i in range(idx, len(sans), 2):
        white = sans[i]
        black = sans[i + 1] if i + 1 < len(sans) else ""
        if black:
            parts.append(f"{move_no}. {white} {black}")
        else:
            parts.append(f"{move_no}. {white}")
        move_no += 1
    return " ".join(parts)


def build_prompt(
    board: chess.Board,
    banter_log: Optional[List[Tuple[int, str, str]]] = None,
    last_move_info: Optional[str] = None,
    last_moves_info: Optional[Dict[str, str]] = None,
    draw_offer: Optional[Dict[str, str]] = None,
    side_override: Optional[str] = None,
) -> str:
    legal_moves = list(board.legal_moves)
    legal_uci = [move.uci() for move in legal_moves]
    legal_san = [board.san(move) for move in legal_moves]
    side = side_override or ("White" if board.turn == chess.WHITE else "Black")
    history = format_history(board)
    ascii_board = board_to_ascii(board)
    table_board = board_to_table(board)
    banter_lines: List[str] = []
    if banter_log:
        for ply, banter_side, text in banter_log[-8:]:
            cleaned = " ".join(text.strip().split())
            if not cleaned:
                continue
            label = banter_side.capitalize() if banter_side else "Unknown"
            banter_lines.append(f"{ply} {label}: {cleaned}")
    banter_block = "\n".join(banter_lines) if banter_lines else "(none)"
    if last_moves_info:
        white_line = last_moves_info.get("white", "(none)")
        black_line = last_moves_info.get("black", "(none)")
        last_line = f"Last move (White): {white_line}\nLast move (Black): {black_line}\n"
    else:
        last_line = f"Last move: {last_move_info}\n" if last_move_info else "Last move: (none)\n"
    draw_line = ""
    if draw_offer and draw_offer.get("status") == "pending":
        by = draw_offer.get("by", "Unknown")
        message = draw_offer.get("message", "(no message)")
        draw_line = (
            f"Draw offer pending from {by}: {message}\n"
            "Respond by calling accept_draw or decline_draw.\n"
        )
    return (
        f"You are playing as {side}.\n"
        f"Board (ASCII):\n{ascii_board}\n"
        f"Board (Table):\n{table_board}\n"
        f"FEN: {board.fen()}\n"
        f"Move history (SAN): {history}\n"
        f"{last_line}"
        f"{draw_line}"
        f"Banter log (shared between both players, keyed by ply id):\n{banter_block}\n"
        f"Legal moves (UCI): {', '.join(legal_uci)}\n"
        f"Legal moves (SAN): {', '.join(legal_san)}\n"
        "A board image is also provided for reference.\n\n"
        "Reply ONLY as JSON with keys: move (UCI) and optional banter.\n"
        "If you need to resign or offer a draw, call the resign or offer_draw tool instead.\n"
        "If a draw is offered to you, respond with accept_draw or decline_draw.\n"
        "Banter should be a short, witty heckle about tactics, blunders, or traps (1 to 2 sentences max).\n"
        "Example: {\"move\":\"e2e4\",\"banter\":\"Walked right into it.\"}\n"
    )


def build_board_image(board: chess.Board, orientation: chess.Color) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        square = 64
        board_size = square * 8
        light = (248, 248, 248)
        dark = (40, 40, 40)
        image = Image.new("RGB", (board_size, board_size), light)
        draw = ImageDraw.Draw(image)
        font = None
        font_paths = [
            str(Path(__file__).resolve().parent / "assets" / "DejaVuSans-Bold.ttf"),
            "DejaVuSans-Bold.ttf",
            "Arial Bold.ttf",
            "Arial.ttf",
        ]
        for font_name in font_paths:
            try:
                font = ImageFont.truetype(font_name, int(square * 0.92))
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        glyph_cache: Dict[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]], Image.Image] = {}

        def render_glyph(
            letter: str,
            fill: Tuple[int, int, int],
            stroke: Tuple[int, int, int],
        ) -> Image.Image:
            key = (letter, fill, stroke)
            cached = glyph_cache.get(key)
            if cached is not None:
                return cached
            canvas_size = square * 2
            canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
            canvas_draw = ImageDraw.Draw(canvas)
            stroke_width = max(2, int(square * 0.1))
            canvas_draw.text(
                (canvas_size / 2, canvas_size / 2),
                letter,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke,
                anchor="mm",
            )
            bbox = canvas.getbbox()
            if not bbox:
                glyph_cache[key] = canvas
                return canvas
            glyph = canvas.crop(bbox)
            target_size = int(square * 0.85)
            scale = target_size / max(1, max(glyph.width, glyph.height))
            new_w = max(1, int(glyph.width * scale))
            new_h = max(1, int(glyph.height * scale))
            glyph = glyph.resize((new_w, new_h), resample=Image.LANCZOS)
            glyph_cache[key] = glyph
            return glyph

        for row in range(8):
            for col in range(8):
                color = light if (row + col) % 2 == 0 else dark
                x0 = col * square
                y0 = row * square
                draw.rectangle([x0, y0, x0 + square, y0 + square], fill=color)
        last_move = board.peek() if board.move_stack else None
        if last_move:
            from_file = chess.square_file(last_move.from_square)
            from_rank = chess.square_rank(last_move.from_square)
            to_file = chess.square_file(last_move.to_square)
            to_rank = chess.square_rank(last_move.to_square)

            if orientation == chess.WHITE:
                from_col, from_row = from_file, 7 - from_rank
                to_col, to_row = to_file, 7 - to_rank
            else:
                from_col, from_row = 7 - from_file, from_rank
                to_col, to_row = 7 - to_file, to_rank

            overlay = Image.new("RGBA", (board_size, board_size), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            highlight = (245, 196, 66, 100)
            for col, row in ((from_col, from_row), (to_col, to_row)):
                x0 = col * square
                y0 = row * square
                overlay_draw.rectangle([x0, y0, x0 + square, y0 + square], fill=highlight)
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(image)

        for row in range(8):
            for col in range(8):
                if orientation == chess.WHITE:
                    square_index = chess.square(col, 7 - row)
                else:
                    square_index = chess.square(7 - col, row)
                x0 = col * square
                y0 = row * square
                piece = board.piece_at(square_index)
                if not piece:
                    continue
                letter = piece.symbol().upper()
                fill = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                stroke = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                glyph = render_glyph(letter, fill, stroke)
                x = int(x0 + (square - glyph.width) / 2)
                y = int(y0 + (square - glyph.height) / 2)
                image.paste(glyph, (x, y), glyph)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return tiny_png_data_url()

def parse_move_text(text: str, board: chess.Board) -> Optional[chess.Move]:
    legal_moves: Dict[str, chess.Move] = {move.uci(): move for move in board.legal_moves}
    move = legal_moves.get(text.strip().lower())
    if move:
        return move
    candidate = text.strip().replace("0", "O")
    try:
        move = board.parse_san(candidate)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def parse_move_reply(text: str, board: chess.Board) -> Tuple[Optional[chess.Move], Optional[str]]:
    banter: Optional[str] = None
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        move_text = payload.get("move")
        if isinstance(payload.get("banter"), str):
            banter = payload.get("banter")
        if isinstance(move_text, str):
            move = parse_move_text(move_text, board)
            if move:
                return move, banter
    return None, banter


def extract_move(text: str, board: chess.Board) -> Optional[chess.Move]:
    legal_moves: Dict[str, chess.Move] = {move.uci(): move for move in board.legal_moves}
    uci_matches = re.findall(r"[a-h][1-8][a-h][1-8][qrbn]?", text.lower())
    for uci in uci_matches:
        move = legal_moves.get(uci)
        if move:
            return move
    tokens = re.findall(r"[A-Za-z0-9=+#O0\-]+", text)
    for token in tokens:
        candidate = token.replace("0", "O")
        try:
            move = board.parse_san(candidate)
        except ValueError:
            continue
        if move in board.legal_moves:
            return move
    return None


def clear_screen(enabled: bool) -> None:
    if not enabled:
        return
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def choose_move(
    client: OpenRouterChessClient,
    board: chess.Board,
    player: PlayerConfig,
    timeout: Optional[float],
    reasoning: Optional[Dict[str, object]],
    banter_log: Optional[List[Tuple[int, str, str]]] = None,
    last_moves_info: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[chess.Move], str, bool, Optional[float], float, Optional[str], Optional[Dict[str, object]]]:
    if last_moves_info is None:
        last_move_info = None
        if board.move_stack:
            last_move_info = f"{board.peek().uci()} (accuracy n/a)"
        prompt = build_prompt(board, banter_log, last_move_info)
    else:
        prompt = build_prompt(board, banter_log, None, last_moves_info)
    board_image = build_board_image(board, board.turn)
    raw, cost, elapsed, tool_action = client.ask_move(
        player.model,
        prompt,
        player.temperature,
        timeout,
        reasoning,
        board_image_url=board_image,
    )
    if tool_action:
        return None, raw.strip(), False, cost, elapsed, None, tool_action
    move, banter = parse_move_reply(raw, board)
    if move is None:
        move = extract_move(raw, board)
    used_fallback = False
    if move is None:
        move = next(iter(board.legal_moves))
        used_fallback = True
    return move, raw.strip(), used_fallback, cost, elapsed, banter, None


def render_state(
    board: chess.Board,
    player_a: PlayerConfig,
    player_b: PlayerConfig,
    last_raw: Optional[str],
    used_fallback: bool,
    clear: bool,
) -> None:
    clear_screen(clear)
    print("LLM Chess Match")
    print(f"White: {player_a.name} ({player_a.model})")
    print(f"Black: {player_b.name} ({player_b.model})")
    print()
    print(board_to_ascii(board))
    print()
    history = format_history(board)
    if history != "(none)":
        print("Moves:")
        print(history)
        print()
    if last_raw is not None:
        print("Last model reply:")
        print(last_raw)
        print()
    if used_fallback:
        print("Notice: model reply was not a legal move. Fallback to first legal move.")
        print()


def play_game(
    client: OpenRouterChessClient,
    white: PlayerConfig,
    black: PlayerConfig,
    max_plies: int,
    request_timeout: Optional[float],
    delay: float,
    render: bool,
    clear: bool,
) -> Tuple[str, Optional[chess.Outcome], Dict[str, Dict[str, object]], Dict[str, Optional[float]], Dict[str, Dict[str, object]]]:
    board = chess.Board()
    engine = create_stockfish()
    engine.set_fen_position(chess.STARTING_FEN)
    last_raw: Optional[str] = None
    used_fallback = False
    banter_log: List[Tuple[int, str, str]] = []
    last_moves_info = {"white": "(none)", "black": "(none)"}
    cost_totals = {white.model: 0.0, black.model: 0.0}
    cost_moves = {white.model: 0, black.model: 0}
    saw_cost = {white.model: False, black.model: False}
    missing_cost = {white.model: False, black.model: False}
    time_totals = {white.model: 0.0, black.model: 0.0}
    time_moves = {white.model: 0, black.model: 0}

    manual_result: Optional[str] = None
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        active = white if board.turn == chess.WHITE else black
        is_white = board.turn == chess.WHITE
        move, last_raw, used_fallback, call_cost, call_time, banter, tool_action = choose_move(
            client,
            board,
            active,
            request_timeout,
            active.reasoning,
            banter_log,
            last_moves_info,
        )
        model_id = active.model
        cost_moves[model_id] += 1
        time_moves[model_id] += 1
        if call_cost is None:
            missing_cost[model_id] = True
        else:
            saw_cost[model_id] = True
            cost_totals[model_id] += call_cost
        time_totals[model_id] += call_time
        if tool_action:
            name = tool_action.get("name")
            args = tool_action.get("arguments") or {}
            reason = args.get("reason") or args.get("message") or ""
            if name == "resign":
                last_raw = f"[resign] {reason}".strip()
                manual_result = "0-1" if is_white else "1-0"
                break
            if name in {"offer_draw", "accept_draw"}:
                last_raw = f"[{name}] {reason}".strip()
                manual_result = "1/2-1/2"
                break
            if name == "decline_draw":
                last_raw = f"[decline_draw] {reason}".strip()
                move = next(iter(board.legal_moves))
                used_fallback = True
        if move is None:
            move = next(iter(board.legal_moves))
            used_fallback = True
        san = board.san(move)
        board.push(move)
        engine.make_moves_from_current_position([move.uci()])
        eval_summary = eval_summary_for_turn(engine.get_evaluation(), board.turn)
        eval_text = format_eval_text(eval_summary)
        last_moves_info["white" if is_white else "black"] = (
            f"{san} ({move.uci()}) | Accuracy n/a | Eval {eval_text}"
        )
        if isinstance(banter, str):
            cleaned_banter = banter.strip()
            if cleaned_banter:
                banter_log.append((len(board.move_stack), "white" if is_white else "black", cleaned_banter))
        if render:
            render_state(board, white, black, last_raw, used_fallback, clear)
            if delay:
                time.sleep(delay)

    cost_report: Dict[str, Dict[str, object]] = {}
    for model_id in cost_totals:
        moves = cost_moves[model_id]
        if moves == 0 or not saw_cost[model_id]:
            cost_report[model_id] = {"avg_cost": None, "status": "unknown"}
            continue
        avg_cost = cost_totals[model_id] / float(moves)
        status = "partial" if missing_cost[model_id] else "complete"
        cost_report[model_id] = {"avg_cost": avg_cost, "status": status}

    time_report: Dict[str, Dict[str, object]] = {}
    for model_id, total in time_totals.items():
        moves = time_moves[model_id]
        if moves == 0:
            time_report[model_id] = {"avg_time": None}
        else:
            time_report[model_id] = {"avg_time": total / float(moves)}

    moves_uci = [move.uci() for move in board.move_stack]
    accuracy_report = compute_accuracy_report(moves_uci, white.model, black.model)

    if manual_result is not None:
        return manual_result, None, cost_report, accuracy_report, time_report
    return board.result(claim_draw=True), board.outcome(claim_draw=True), cost_report, accuracy_report, time_report


def run_headless_series(
    client: OpenRouterChessClient,
    models: List[ModelSpec],
    games: int,
    max_plies: int,
    request_timeout: Optional[float],
) -> None:
    if games < 1:
        raise SystemExit("--games must be at least 1 in headless mode.")
    if len(models) < 2:
        raise SystemExit("Headless mode requires at least two models in the pool.")
    path = leaderboard_path()
    players = load_leaderboard(path)
    scores: Dict[str, float] = {}
    games_played: Dict[str, int] = {}
    scheduled_games: Dict[str, int] = {}
    max_workers = 10
    futures = {}

    def adjust_scheduled(model_id: str, delta: int) -> None:
        scheduled_games[model_id] = scheduled_games.get(model_id, 0) + delta
        if scheduled_games[model_id] <= 0:
            scheduled_games.pop(model_id, None)

    def schedule_game(game_index: int, executor: ThreadPoolExecutor) -> None:
        model_a, model_b = choose_model_pair(models, players, scheduled_games)
        if random.choice([True, False]):
            white_spec, black_spec = model_a, model_b
        else:
            white_spec, black_spec = model_b, model_a

        white = PlayerConfig(
            model=white_spec.model,
            temperature=white_spec.temperature,
            name=white_spec.model,
            reasoning=white_spec.reasoning,
        )
        black = PlayerConfig(
            model=black_spec.model,
            temperature=black_spec.temperature,
            name=black_spec.model,
            reasoning=black_spec.reasoning,
        )

        adjust_scheduled(white.model, 1)
        adjust_scheduled(black.model, 1)
        print(f"Game {game_index}: White {white.model} vs Black {black.model}")
        future = executor.submit(
            play_game,
            client,
            white,
            black,
            max_plies,
            request_timeout,
            0.0,
            False,
            False,
        )
        futures[future] = (game_index, white, black)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        next_game = 1
        while next_game <= games or futures:
            while next_game <= games and len(futures) < max_workers:
                schedule_game(next_game, executor)
                next_game += 1

            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                game_index, white, black = futures.pop(future)
                adjust_scheduled(white.model, -1)
                adjust_scheduled(black.model, -1)
                try:
                    result, _, cost_report, accuracy_report, time_report = future.result()
                except Exception as exc:
                    print(f"Game {game_index} failed: {exc}")
                    continue

                def format_cost(entry: Dict[str, object]) -> str:
                    avg = entry.get("avg_cost")
                    status = entry.get("status")
                    if avg is None or status == "unknown":
                        return "n/a"
                    note = f"${float(avg):.6f}"
                    if status == "partial":
                        note += " (partial)"
                    return note

                def format_time(entry: Dict[str, object]) -> str:
                    avg = entry.get("avg_time")
                    if avg is None:
                        return "n/a"
                    return f"{float(avg):.2f}s"

                white_cost = format_cost(cost_report.get(white.model, {}))
                black_cost = format_cost(cost_report.get(black.model, {}))
                white_time = format_time(time_report.get(white.model, {}))
                black_time = format_time(time_report.get(black.model, {}))
                print(
                    f"Result: {result} | Avg Cost/Move: W {white_cost} | B {black_cost} | "
                    f"Avg Time/Move: W {white_time} | B {black_time}"
                )

                apply_result(players, white.model, black.model, result)
                for model_id, entry in cost_report.items():
                    if entry.get("status") == "complete" and entry.get("avg_cost") is not None:
                        update_cost_average(players, model_id, float(entry["avg_cost"]))
                for model_id, accuracy in accuracy_report.items():
                    if accuracy is not None:
                        update_accuracy_average(players, model_id, float(accuracy))
                for model_id, entry in time_report.items():
                    avg_time = entry.get("avg_time")
                    if avg_time is not None:
                        update_time_average(players, model_id, float(avg_time))
                save_leaderboard(path, players)

                for model_id in (white.model, black.model):
                    games_played[model_id] = games_played.get(model_id, 0) + 1
                    scores.setdefault(model_id, 0.0)

                if result == "1-0":
                    scores[white.model] += 1.0
                elif result == "0-1":
                    scores[black.model] += 1.0
                else:
                    scores[white.model] += 0.5
                    scores[black.model] += 0.5

    print("Headless match complete.")
    print(f"Games played: {games}")
    if scores:
        print("Series scores:")
        for model_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            played = games_played.get(model_id, 0)
            print(f"{model_id}: {score:.1f} ({played} games)")
    print_leaderboard()


def run_match(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in the environment.")
    if args.seed is not None:
        random.seed(args.seed)

    client = OpenRouterChessClient(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        app_url=args.app_url,
        app_title=args.app_title,
    )

    if args.headless:
        model_pool = load_model_pool(models_path())
        run_headless_series(
            client,
            models=model_pool,
            games=args.games,
            max_plies=args.max_plies,
            request_timeout=args.request_timeout,
        )
        return

    reasoning_config_a = build_reasoning_config(
        args.no_reasoning,
        args.reasoning_effort_a,
        args.reasoning_max_tokens_a,
        args.reasoning_include,
    )
    reasoning_config_b = build_reasoning_config(
        args.no_reasoning,
        args.reasoning_effort_b,
        args.reasoning_max_tokens_b,
        args.reasoning_include,
    )

    player_a = PlayerConfig(
        model=args.model_a,
        temperature=args.temperature_a,
        name="Model A",
        reasoning=reasoning_config_a,
    )
    player_b = PlayerConfig(
        model=args.model_b,
        temperature=args.temperature_b,
        name="Model B",
        reasoning=reasoning_config_b,
    )

    result, outcome, cost_report, accuracy_report, time_report = play_game(
        client,
        white=player_a,
        black=player_b,
        max_plies=args.max_plies,
        request_timeout=args.request_timeout,
        delay=args.delay,
        render=True,
        clear=not args.no_clear,
    )
    print("Game over.")
    print(f"Result: {result}")
    def format_cost(entry: Dict[str, object]) -> str:
        avg = entry.get("avg_cost")
        status = entry.get("status")
        if avg is None or status == "unknown":
            return "n/a"
        note = f"${float(avg):.6f}"
        if status == "partial":
            note += " (partial)"
        return note
    def format_time(entry: Dict[str, object]) -> str:
        avg = entry.get("avg_time")
        if avg is None:
            return "n/a"
        return f"{float(avg):.2f}s"
    white_cost = format_cost(cost_report.get(player_a.model, {}))
    black_cost = format_cost(cost_report.get(player_b.model, {}))
    print(f"Avg Cost/Move: W {white_cost} | B {black_cost}")
    white_time = format_time(time_report.get(player_a.model, {}))
    black_time = format_time(time_report.get(player_b.model, {}))
    print(f"Avg Time/Move: W {white_time} | B {black_time}")
    print(outcome)
    path = leaderboard_path()
    players = load_leaderboard(path)
    apply_result(players, player_a.model, player_b.model, result)
    for model_id, entry in cost_report.items():
        if entry.get("status") == "complete" and entry.get("avg_cost") is not None:
            update_cost_average(players, model_id, float(entry["avg_cost"]))
    for model_id, accuracy in accuracy_report.items():
        if accuracy is not None:
            update_accuracy_average(players, model_id, float(accuracy))
    for model_id, entry in time_report.items():
        avg_time = entry.get("avg_time")
        if avg_time is not None:
            update_time_average(players, model_id, float(avg_time))
    save_leaderboard(path, players)
    print_leaderboard()


def main() -> None:
    args = parse_args()
    run_match(args)


if __name__ == "__main__":
    main()
