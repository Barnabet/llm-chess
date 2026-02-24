import json
import os
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Deque, Dict, List, Optional, Tuple

import chess
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from stockfish import Stockfish

from llm_chess.cli import (
    ModelSpec,
    OpenRouterChessClient,
    accuracy_from_losses,
    apply_result,
    build_board_image,
    build_prompt,
    build_reasoning_from_spec,
    create_stockfish,
    eval_to_cp,
    eval_to_cp_for_turn,
    format_eval_text,
    extract_move,
    get_starting_elo,
    parse_move_reply,
    leaderboard_path,
    load_leaderboard,
    models_path,
    save_leaderboard,
    update_accuracy_average,
    update_cost_average,
    update_time_average,
)


HUMAN_MODEL_ID = "human/local"
DEFAULT_MAX_PLIES = 120
DEFAULT_DELAY = 0.4
OPENING_DELAY = 0.35
OPENING_MODEL_ID = "opening/book"
NON_TRACKED_MODELS = {HUMAN_MODEL_ID, OPENING_MODEL_ID}

# Stockfish engine players at different ELO levels
# Note: Stockfish library minimum ELO is 1320
STOCKFISH_ELO_LEVELS = [1350, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
STOCKFISH_MODELS = {f"stockfish/elo-{elo}": elo for elo in STOCKFISH_ELO_LEVELS}
STOCKFISH_MODEL_IDS = set(STOCKFISH_MODELS.keys())

OPENING_LIBRARY = [
    {"name": "Sicilian Defense", "white": "e4", "black": "c5"},
    {"name": "French Defense", "white": "e4", "black": "e6"},
    {"name": "Caro-Kann Defense", "white": "e4", "black": "c6"},
    {"name": "Alekhine Defense", "white": "e4", "black": "Nf6"},
    {"name": "Scandinavian Defense", "white": "e4", "black": "d5"},
    {"name": "Pirc Defense", "white": "e4", "black": "d6"},
    {"name": "Modern Defense", "white": "e4", "black": "g6"},
    {"name": "Open Game", "white": "e4", "black": "e5"},
    {"name": "Queen's Pawn Game", "white": "d4", "black": "d5"},
    {"name": "Indian Defense", "white": "d4", "black": "Nf6"},
    {"name": "Dutch Defense", "white": "d4", "black": "f5"},
    {"name": "Old Benoni", "white": "d4", "black": "c5"},
    {"name": "English (Rev. Sicilian)", "white": "c4", "black": "e5"},
    {"name": "English (Symmetrical)", "white": "c4", "black": "c5"},
    {"name": "Reti Opening", "white": "Nf3", "black": "d5"},
    {"name": "Bird's Opening", "white": "f4", "black": "d5"},
    {"name": "Nimzo-Larsen Attack", "white": "b3", "black": "d5"},
]


@dataclass
class GamePlayer:
    kind: str
    model: str
    name: str
    temperature: float
    reasoning: Optional[Dict[str, object]]
    color: bool


@dataclass
class MoveEntry:
    ply: int
    uci: str
    san: str
    side: str
    model: str
    loss: Optional[int]
    accuracy: Optional[float]
    banter: Optional[str]
    eval_type: Optional[str]
    eval_value: Optional[float]
    eval_cp: Optional[int]


@dataclass
class GameState:
    game_id: str
    mode: str
    white: GamePlayer
    black: GamePlayer
    board: chess.Board = field(default_factory=chess.Board)
    moves: List[MoveEntry] = field(default_factory=list)
    max_plies: int = DEFAULT_MAX_PLIES
    delay: float = DEFAULT_DELAY
    created_at: float = field(default_factory=time.time)
    result: Optional[str] = None
    outcome: Optional[chess.Outcome] = None
    finished: bool = False
    error: Optional[str] = None
    last_raw: Optional[str] = None
    last_fallback: bool = False
    thinking: bool = False
    lock: Lock = field(default_factory=Lock)
    draw_offer: Optional[Dict[str, object]] = None
    last_tool: Optional[Dict[str, object]] = None
    end_reason: Optional[str] = None
    engine: Stockfish = field(default_factory=create_stockfish)
    cost_totals: Dict[str, float] = field(default_factory=dict)
    cost_moves: Dict[str, int] = field(default_factory=dict)
    saw_cost: Dict[str, bool] = field(default_factory=dict)
    missing_cost: Dict[str, bool] = field(default_factory=dict)
    time_totals: Dict[str, float] = field(default_factory=dict)
    time_moves: Dict[str, int] = field(default_factory=dict)
    premove_evals: Dict[str, Deque[int]] = field(
        default_factory=lambda: {"white": deque(maxlen=1), "black": deque(maxlen=1)}
    )


GAMES: Dict[str, GameState] = {}
GAMES_LOCK = Lock()
SUBSCRIBERS: Dict[str, List[Queue]] = {}
SUBSCRIBERS_LOCK = Lock()


def asset_root() -> Path:
    return Path(__file__).resolve().parent / "web"


def load_model_catalog(path: Path) -> Dict[str, ModelSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"{path} must contain a top-level 'models' list.")
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("Models list must be an array.")
    models: Dict[str, ModelSpec] = {}
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model") or item.get("id")
        if not model_id:
            continue
        temperature = float(item.get("temperature", 0.2))
        try:
            reasoning = build_reasoning_from_spec(item)
        except SystemExit as exc:
            raise ValueError(str(exc)) from exc
        models[model_id] = ModelSpec(model=model_id, temperature=temperature, reasoning=reasoning)
    return models


def find_game(game_id: str) -> Optional[GameState]:
    with GAMES_LOCK:
        return GAMES.get(game_id)


def get_game(game_id: str) -> GameState:
    game = find_game(game_id)
    if not game:
        abort(404, description="Game not found")
    return game


def subscribe_game(game_id: str) -> Queue:
    queue: Queue = Queue(maxsize=32)
    with SUBSCRIBERS_LOCK:
        SUBSCRIBERS.setdefault(game_id, []).append(queue)
    return queue


def unsubscribe_game(game_id: str, queue: Queue) -> None:
    with SUBSCRIBERS_LOCK:
        queues = SUBSCRIBERS.get(game_id)
        if not queues:
            return
        if queue in queues:
            queues.remove(queue)
        if not queues:
            SUBSCRIBERS.pop(game_id, None)


def publish_game(game: GameState) -> None:
    with SUBSCRIBERS_LOCK:
        queues = list(SUBSCRIBERS.get(game.game_id, []))
    if not queues:
        return
    snapshot = game_snapshot(game)
    for queue in queues:
        try:
            queue.put_nowait(snapshot)
        except Full:
            continue


def side_name(turn: bool) -> str:
    return "white" if turn == chess.WHITE else "black"


def player_for_turn(game: GameState) -> GamePlayer:
    return game.white if game.board.turn == chess.WHITE else game.black


def should_track_model(model_id: str) -> bool:
    return model_id not in NON_TRACKED_MODELS


def should_track_cost_time(model_id: str) -> bool:
    """Return True if model should track cost and time metrics (excludes Stockfish)."""
    if model_id in NON_TRACKED_MODELS:
        return False
    if model_id in STOCKFISH_MODEL_IDS:
        return False
    return True


def cost_report(game: GameState) -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    for model_id, total in game.cost_totals.items():
        moves = game.cost_moves.get(model_id, 0)
        saw_cost = game.saw_cost.get(model_id, False)
        missing = game.missing_cost.get(model_id, False)
        if moves == 0 or not saw_cost:
            report[model_id] = {"avg_cost": None, "status": "unknown"}
        else:
            report[model_id] = {
                "avg_cost": total / float(moves),
                "status": "partial" if missing else "complete",
            }
    return report


def time_report(game: GameState) -> Dict[str, Dict[str, object]]:
    report: Dict[str, Dict[str, object]] = {}
    for model_id, total in game.time_totals.items():
        moves = game.time_moves.get(model_id, 0)
        if moves == 0:
            report[model_id] = {"avg_time": None}
        else:
            report[model_id] = {"avg_time": total / float(moves)}
    return report


def accuracy_summary(game: GameState) -> Dict[str, Dict[str, Optional[float]]]:
    losses: Dict[str, List[int]] = {}
    for move in game.moves:
        if move.loss is None:
            continue
        losses.setdefault(move.model, []).append(move.loss)
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for model_id, loss_list in losses.items():
        avg = accuracy_from_losses(loss_list)
        summary[model_id] = {"avg": avg, "moves": float(len(loss_list))}
    return summary


def current_evaluation(game: GameState) -> Dict[str, Optional[float]]:
    try:
        evaluation = game.engine.get_evaluation()
    except Exception:
        return {"type": "cp", "value": 0, "cp": 0}
    if not isinstance(evaluation, dict):
        return {"type": "cp", "value": 0, "cp": 0}
    value = evaluation.get("value")
    eval_type = evaluation.get("type", "cp")
    try:
        cp = eval_to_cp(evaluation)
    except (TypeError, ValueError):
        cp = 0
    if game.board.turn == chess.BLACK:
        cp = -cp
        if isinstance(value, (int, float)):
            value = -value
    return {"type": eval_type, "value": value, "cp": cp}


def record_move(
    game: GameState,
    move: chess.Move,
    model_id: str,
    raw: Optional[str],
    fallback: bool,
    banter: Optional[str],
) -> None:
    is_white = game.board.turn == chess.WHITE
    cleaned_banter = banter.strip() if isinstance(banter, str) else None
    color_key = "white" if is_white else "black"
    first_move = not any(entry.side == color_key for entry in game.moves)
    premove_eval = eval_to_cp_for_turn(game.engine.get_evaluation(), game.board.turn)
    premove_queue = game.premove_evals[color_key]
    premove_queue.clear()
    premove_queue.append(premove_eval)
    baseline = premove_queue[0]
    san = game.board.san(move)
    game.board.push(move)
    game.engine.make_moves_from_current_position([move.uci()])
    post_eval = eval_to_cp_for_turn(game.engine.get_evaluation(), game.board.turn)
    eval_summary = current_evaluation(game)
    if first_move:
        loss = 0
        accuracy = 100.0
    else:
        # Positive loss = position worsened, negative loss = position improved
        loss = baseline - post_eval if is_white else post_eval - baseline
        # Allow accuracy > 100% for moves that improve position
        # Negative loss results in accuracy above 100%
        accuracy = max(0.0, 100.0 - (loss / 1.5))
    game.moves.append(
        MoveEntry(
            ply=len(game.board.move_stack),
            uci=move.uci(),
            san=san,
            side="white" if is_white else "black",
            model=model_id,
            loss=loss,
            accuracy=accuracy,
            banter=cleaned_banter,
            eval_type=eval_summary.get("type"),
            eval_value=eval_summary.get("value"),
            eval_cp=eval_summary.get("cp"),
        )
    )
    game.last_raw = raw
    game.last_fallback = fallback
    publish_game(game)


def end_reason_from_outcome(result: str, outcome: Optional[chess.Outcome]) -> str:
    if outcome and outcome.termination:
        termination = outcome.termination
        winner = outcome.winner
        if termination == chess.Termination.CHECKMATE:
            return f"{'White' if winner == chess.WHITE else 'Black'} wins by checkmate"
        if termination == chess.Termination.STALEMATE:
            return "Draw by stalemate"
        if termination == chess.Termination.INSUFFICIENT_MATERIAL:
            return "Draw by insufficient material"
        if termination == chess.Termination.SEVENTYFIVE_MOVES:
            return "Draw by seventy-five move rule"
        if termination == chess.Termination.FIVEFOLD_REPETITION:
            return "Draw by fivefold repetition"
        if termination == chess.Termination.FIFTY_MOVES:
            return "Draw by fifty-move rule"
        if termination == chess.Termination.THREEFOLD_REPETITION:
            return "Draw by threefold repetition"
        if termination == chess.Termination.TIMEOUT:
            if winner is None:
                return "Draw by timeout"
            return f"{'White' if winner == chess.WHITE else 'Black'} wins on time"
    if result == "1-0":
        return "White wins"
    if result == "0-1":
        return "Black wins"
    return "Draw"


def finalize_game(
    game: GameState,
    result: str,
    outcome: Optional[chess.Outcome] = None,
    reason: Optional[str] = None,
) -> None:
    game.finished = True
    game.result = result
    game.outcome = outcome
    game.end_reason = reason or end_reason_from_outcome(result, outcome)
    track_leaderboard = game.white.kind != "human" and game.black.kind != "human"
    if track_leaderboard:
        players = load_leaderboard(leaderboard_path())
        for model_id in NON_TRACKED_MODELS:
            players.pop(model_id, None)
        # Debug logging for game result tracking
        print(f"[DEBUG finalize_game] result={result}, reason={reason}")
        print(f"[DEBUG finalize_game] white={game.white.model}, black={game.black.model}")
        apply_result(players, game.white.model, game.black.model, result)

        # Only update cost for LLM models (not Stockfish)
        for model_id, entry in cost_report(game).items():
            if not should_track_cost_time(model_id):
                continue
            if entry.get("status") == "complete" and entry.get("avg_cost") is not None:
                update_cost_average(players, model_id, float(entry["avg_cost"]))

        # Update accuracy for ALL tracked models (including Stockfish)
        for model_id, stats in accuracy_summary(game).items():
            if not should_track_model(model_id):
                continue
            avg = stats.get("avg")
            if avg is not None:
                update_accuracy_average(players, model_id, float(avg))

        # Only update time for LLM models (not Stockfish)
        for model_id, entry in time_report(game).items():
            if not should_track_cost_time(model_id):
                continue
            avg_time = entry.get("avg_time")
            if avg_time is not None:
                update_time_average(players, model_id, float(avg_time))

        save_leaderboard(leaderboard_path(), players)
    publish_game(game)


def maybe_finalize_game(game: GameState) -> None:
    if game.finished or game.error:
        return
    if not game.board.is_game_over(claim_draw=True) and len(game.board.move_stack) < game.max_plies:
        return
    finalize_game(game, game.board.result(claim_draw=True), game.board.outcome(claim_draw=True))


def perform_stockfish_move(
    game: GameState,
    player: GamePlayer,
    target_elo: int,
) -> None:
    """Generate a move using Stockfish at a specific ELO level."""
    with game.lock:
        if game.finished or game.error:
            return
        if game.board.turn != player.color:
            return

        # Create a temporary Stockfish instance with ELO limiting
        engine = create_stockfish()
        engine.set_fen_position(game.board.fen())
        engine.set_depth(20)
        engine.set_elo_rating(target_elo)

        # Get best move from Stockfish
        best_move_uci = engine.get_best_move()
        if best_move_uci is None:
            # Fallback to first legal move if Stockfish returns None
            move = next(iter(game.board.legal_moves))
        else:
            move = chess.Move.from_uci(best_move_uci)

        record_move(game, move, player.model, None, False, None)
        maybe_finalize_game(game)

    if game.delay:
        time.sleep(game.delay)


def perform_llm_move(
    game: GameState,
    player: GamePlayer,
    client: OpenRouterChessClient,
    timeout: Optional[float],
) -> None:
    draw_response_only = False
    with game.lock:
        if game.finished or game.error:
            return
        draw_response_only = False
        if game.draw_offer and game.draw_offer.get("status") == "pending":
            offerer = game.draw_offer.get("by")
            if offerer != side_name(player.color):
                draw_response_only = True
        if game.board.turn != player.color and not draw_response_only:
            return
        banter_history = [(move.ply, move.side, move.banter) for move in game.moves if move.banter]
        last_white = next((move for move in reversed(game.moves) if move.side == "white"), None)
        last_black = next((move for move in reversed(game.moves) if move.side == "black"), None)

        def move_summary(entry: Optional[MoveEntry]) -> str:
            if not entry:
                return "(none)"
            acc = entry.accuracy
            acc_text = f"{acc:.1f}%" if isinstance(acc, (int, float)) else "n/a"
            eval_summary = {
                "type": entry.eval_type,
                "value": entry.eval_value,
                "cp": entry.eval_cp,
            }
            eval_text = format_eval_text(eval_summary)
            return f"{entry.san} ({entry.uci}) | Accuracy {acc_text} | Eval {eval_text}"

        last_moves_info = {
            "white": move_summary(last_white),
            "black": move_summary(last_black),
        }
        pending_offer = None
        if game.draw_offer and game.draw_offer.get("status") == "pending":
            if game.draw_offer.get("by") != side_name(player.color):
                pending_offer = {
                    "by": game.draw_offer.get("by"),
                    "message": game.draw_offer.get("message"),
                    "status": "pending",
                }
        side_override = "White" if player.color == chess.WHITE else "Black"
        prompt = build_prompt(game.board, banter_history, None, last_moves_info, pending_offer, side_override)
        board_image = build_board_image(game.board, player.color)
        game.thinking = True
    raw = ""
    fallback = False
    cost = None
    call_time: Optional[float] = None
    tool_action: Optional[Dict[str, object]] = None
    banter: Optional[str] = None
    try:
        raw, cost, call_time, tool_action = client.ask_move(
            player.model,
            prompt,
            player.temperature,
            timeout,
            player.reasoning,
            board_image_url=board_image,
        )
    except Exception as exc:
        with game.lock:
            game.error = str(exc)
            game.finished = True
            game.thinking = False
            publish_game(game)
        return

    with game.lock:
        game.thinking = False
        if game.finished or game.error:
            return
        if game.board.turn != player.color and not draw_response_only:
            return
        if cost is None:
            game.missing_cost[player.model] = True
        else:
            game.saw_cost[player.model] = True
            game.cost_totals[player.model] = game.cost_totals.get(player.model, 0.0) + float(cost)
        game.cost_moves[player.model] = game.cost_moves.get(player.model, 0) + 1
        if call_time is not None:
            game.time_totals[player.model] = game.time_totals.get(player.model, 0.0) + float(call_time)
            game.time_moves[player.model] = game.time_moves.get(player.model, 0) + 1
        if tool_action:
            name = tool_action.get("name")
            args = tool_action.get("arguments") or {}
            reason = args.get("reason") or args.get("message") or ""
            if name == "resign":
                game.last_raw = f"[resign] {reason}".strip()
                game.last_tool = {"name": "resign", "by": side_name(player.color), "reason": reason}
                winner = "Black" if player.color == chess.WHITE else "White"
                resign_result = "0-1" if player.color == chess.WHITE else "1-0"
                print(f"[DEBUG resign] player.model={player.model}, player.color={'WHITE' if player.color == chess.WHITE else 'BLACK'}")
                print(f"[DEBUG resign] winner={winner}, result={resign_result}")
                finalize_game(game, resign_result, reason=f"{winner} wins by resignation")
                return
            if name == "offer_draw":
                game.last_raw = f"[offer_draw] {reason}".strip()
                game.last_tool = {"name": "offer_draw", "by": side_name(player.color), "message": reason}
                game.draw_offer = {
                    "by": side_name(player.color),
                    "message": reason,
                    "status": "pending",
                    "response_by": None,
                    "response": None,
                }
                publish_game(game)
                return
            if name == "accept_draw":
                if game.draw_offer and game.draw_offer.get("status") == "pending":
                    game.last_raw = f"[accept_draw] {reason}".strip()
                    game.last_tool = {"name": "accept_draw", "by": side_name(player.color), "message": reason}
                    game.draw_offer["status"] = "accepted"
                    game.draw_offer["response_by"] = side_name(player.color)
                    game.draw_offer["response"] = reason
                    finalize_game(game, "1/2-1/2", reason="Draw agreed")
                else:
                    game.last_raw = "[accept_draw] (no pending draw offer)"
                    publish_game(game)
                return
            if name == "decline_draw":
                if game.draw_offer and game.draw_offer.get("status") == "pending":
                    game.last_raw = f"[decline_draw] {reason}".strip()
                    game.last_tool = {"name": "decline_draw", "by": side_name(player.color), "message": reason}
                    game.draw_offer["status"] = "declined"
                    game.draw_offer["response_by"] = side_name(player.color)
                    game.draw_offer["response"] = reason
                    publish_game(game)
                else:
                    game.last_raw = "[decline_draw] (no pending draw offer)"
                    publish_game(game)
                return
        if draw_response_only:
            if game.draw_offer and game.draw_offer.get("status") == "pending":
                game.last_raw = "[draw_offer] no response provided"
                game.last_tool = {"name": "decline_draw", "by": side_name(player.color), "message": ""}
                game.draw_offer["status"] = "declined"
                game.draw_offer["response_by"] = side_name(player.color)
                game.draw_offer["response"] = "No response; play continues."
                publish_game(game)
            return

        move, banter = parse_move_reply(raw, game.board)
        if move is None:
            move = extract_move(raw, game.board)
        if move is None:
            move = next(iter(game.board.legal_moves))
            fallback = True
        record_move(game, move, player.model, raw.strip() if raw else None, fallback, banter)
        maybe_finalize_game(game)

    if game.delay:
        time.sleep(game.delay)


def run_game_loop(game_id: str, client: Optional[OpenRouterChessClient], timeout: Optional[float]) -> None:
    """Main game loop that handles LLM and Stockfish players."""
    game = find_game(game_id)
    if not game:
        return
    while True:
        with game.lock:
            if game.finished or game.error:
                break
            if game.board.is_game_over(claim_draw=True) or len(game.board.move_stack) >= game.max_plies:
                maybe_finalize_game(game)
                break
            active = player_for_turn(game)
            if game.draw_offer and game.draw_offer.get("status") == "pending":
                offerer = game.draw_offer.get("by")
                # Handle draw offers for stockfish (auto-decline)
                if active.kind == "stockfish":
                    game.draw_offer["status"] = "declined"
                    game.draw_offer["response_by"] = side_name(active.color)
                    game.draw_offer["response"] = "Stockfish declines draw offers."
                    game.last_tool = {"name": "decline_draw", "by": side_name(active.color), "message": "Stockfish declines."}
                    publish_game(game)
                    continue
                if offerer == side_name(game.board.turn):
                    active = game.black if offerer == "white" else game.white
            if active.kind == "human":
                if game.draw_offer and game.draw_offer.get("status") == "pending":
                    game.draw_offer["status"] = "declined"
                    game.draw_offer["response_by"] = "human"
                    game.draw_offer["response"] = "Human play continues."
                    game.last_tool = {"name": "decline_draw", "by": "human", "message": "Human play continues."}
                    publish_game(game)
                break

        if active.kind == "stockfish":
            target_elo = STOCKFISH_MODELS.get(active.model, 1500)
            perform_stockfish_move(game, active, target_elo)
        elif active.kind == "llm":
            perform_llm_move(game, active, client, timeout)


# Keep alias for backward compatibility
def run_llm_loop(game_id: str, client: OpenRouterChessClient, timeout: Optional[float]) -> None:
    run_game_loop(game_id, client, timeout)


def apply_opening(game: GameState, opening: Dict[str, str], delay: float) -> None:
    moves = [opening.get("white"), opening.get("black")]
    for san in moves:
        if not san:
            continue
        with game.lock:
            if game.finished or game.error:
                return
            try:
                move = game.board.parse_san(san)
            except ValueError:
                game.error = f"Invalid opening move: {san}"
                game.finished = True
                publish_game(game)
                return
            if move not in game.board.legal_moves:
                game.error = f"Illegal opening move: {san}"
                game.finished = True
                publish_game(game)
                return
            record_move(game, move, OPENING_MODEL_ID, None, False, None)
            maybe_finalize_game(game)
        if delay:
            time.sleep(delay)


def run_opening_then_loop(
    game_id: str,
    opening: Dict[str, str],
    client: Optional[OpenRouterChessClient],
    timeout: Optional[float],
    delay: float,
) -> None:
    game = find_game(game_id)
    if not game:
        print(f"[DEBUG run_opening_then_loop] game not found: {game_id}")
        return
    try:
        apply_opening(game, opening, delay)
    except Exception as exc:
        print(f"[DEBUG run_opening_then_loop] apply_opening failed: {exc}")
        with game.lock:
            game.error = f"Opening failed: {exc}"
            game.finished = True
            publish_game(game)
        return
    with game.lock:
        if game.finished or game.error:
            print(f"[DEBUG run_opening_then_loop] game finished/error after opening: finished={game.finished}, error={game.error}")
            return
    run_game_loop(game_id, client, timeout)


def game_snapshot(game: GameState) -> Dict[str, object]:
    players = load_leaderboard(leaderboard_path())
    white_stats = players.get(game.white.model, {}) if players else {}
    black_stats = players.get(game.black.model, {}) if players else {}
    moves = [
        {
            "ply": move.ply,
            "uci": move.uci,
            "san": move.san,
            "side": move.side,
            "model": move.model,
            "loss": move.loss,
            "accuracy": move.accuracy,
            "banter": move.banter,
            "eval_type": move.eval_type,
            "eval_value": move.eval_value,
            "eval_cp": move.eval_cp,
        }
        for move in game.moves
    ]
    last_move = moves[-1] if moves else None
    summary = accuracy_summary(game)
    return {
        "id": game.game_id,
        "mode": game.mode,
        "fen": game.board.fen(),
        "turn": side_name(game.board.turn),
        "white": {
            "name": game.white.name,
            "model": game.white.model,
            "kind": game.white.kind,
            "elo": white_stats.get("elo"),
        },
        "black": {
            "name": game.black.name,
            "model": game.black.model,
            "kind": game.black.kind,
            "elo": black_stats.get("elo"),
        },
        "humanColor": side_name(game.white.color)
        if game.white.kind == "human"
        else (side_name(game.black.color) if game.black.kind == "human" else None),
        "moves": moves,
        "lastMove": {
            "uci": last_move["uci"],
            "side": last_move["side"],
        }
        if last_move
        else None,
        "result": game.result,
        "outcome": str(game.outcome) if game.outcome else None,
        "finished": game.finished,
        "error": game.error,
        "thinking": game.thinking,
        "lastRaw": game.last_raw,
        "lastFallback": game.last_fallback,
        "lastTool": game.last_tool,
        "drawOffer": game.draw_offer,
        "endReason": game.end_reason,
        "cost": cost_report(game),
        "accuracySummary": summary,
        "evaluation": current_evaluation(game),
    }


def get_player_spec(
    model_id: str,
    catalog: Dict[str, ModelSpec],
) -> Optional[Tuple[str, float, Optional[Dict[str, object]], str]]:
    """Get player specification from model ID.

    Returns (model, temperature, reasoning, kind) tuple, or None if not found.
    """
    if model_id in STOCKFISH_MODEL_IDS:
        return model_id, 0.0, None, "stockfish"
    spec = catalog.get(model_id)
    if not spec:
        return None
    return spec.model, spec.temperature, spec.reasoning, "llm"


def create_app() -> Flask:
    app_root = asset_root()
    app = Flask(
        __name__,
        static_folder=str(app_root / "static"),
        static_url_path="/static",
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = None
    if api_key:
        client = OpenRouterChessClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            app_url=os.getenv("OPENROUTER_APP_URL"),
            app_title=os.getenv("OPENROUTER_APP_TITLE"),
        )

    @app.route("/")
    def index() -> object:
        return send_from_directory(app_root, "index.html")

    @app.route("/leaderboard")
    def leaderboard() -> object:
        return send_from_directory(app_root, "leaderboard.html")

    @app.route("/pieces")
    def pieces() -> object:
        return send_from_directory(app_root, "pieces.html")

    @app.route("/.llm_chess/<path:filename>")
    def llm_chess_data(filename: str) -> object:
        data_root = Path.cwd() / ".llm_chess"
        return send_from_directory(data_root, filename)

    @app.route("/piece/<style>/<path:filename>")
    def piece_asset(style: str, filename: str) -> object:
        piece_root = app_root / "piece" / style
        return send_from_directory(piece_root, filename)

    @app.route("/api/models")
    def api_models() -> object:
        try:
            catalog = load_model_catalog(models_path())
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc), "models": []}), 404

        # Build LLM models list
        models_list = [
            {
                "model": spec.model,
                "temperature": spec.temperature,
                "reasoning": spec.reasoning,
                "kind": "llm",
            }
            for spec in catalog.values()
        ]

        # Add Stockfish players (sorted by ELO)
        for model_id, elo in sorted(STOCKFISH_MODELS.items(), key=lambda x: x[1]):
            models_list.append({
                "model": model_id,
                "temperature": 0.0,
                "reasoning": None,
                "kind": "stockfish",
                "starting_elo": elo,
            })

        return jsonify({"models": models_list})

    @app.route("/api/piece-styles")
    def api_piece_styles() -> object:
        piece_root = app_root / "piece"
        if not piece_root.exists():
            return jsonify({"styles": []})
        styles = sorted([path.name for path in piece_root.iterdir() if path.is_dir()])
        return jsonify({"styles": styles})

    @app.route("/api/leaderboard")
    def api_leaderboard() -> object:
        players = load_leaderboard(leaderboard_path())
        for model_id in NON_TRACKED_MODELS:
            players.pop(model_id, None)
        return jsonify({"players": players})

    @app.route("/api/game", methods=["POST"])
    def api_game_create() -> object:
        payload = request.get_json(silent=True) or {}
        mode = payload.get("mode")
        opening_choice: Optional[str] = None
        try:
            catalog = load_model_catalog(models_path())
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 404

        max_plies = int(payload.get("maxPlies") or DEFAULT_MAX_PLIES)
        delay = float(payload.get("delay") or DEFAULT_DELAY)
        game_id = uuid.uuid4().hex
        opponent_player: Optional[GamePlayer] = None
        try:
            if mode == "human_vs_llm":
                model_id = payload.get("model")
                human_color = payload.get("humanColor", "white")
                if human_color not in ("white", "black"):
                    return jsonify({"error": "humanColor must be 'white' or 'black'."}), 400
                player_spec = get_player_spec(model_id, catalog)
                if player_spec is None:
                    return jsonify({"error": "Model not found."}), 404
                model, temp, reasoning, kind = player_spec
                # Check if LLM and no API key
                if kind == "llm" and not client:
                    return jsonify({"error": "OPENROUTER_API_KEY is not set."}), 400
                opponent_player = GamePlayer(
                    kind=kind,
                    model=model,
                    name=model,
                    temperature=temp,
                    reasoning=reasoning,
                    color=chess.WHITE if human_color == "black" else chess.BLACK,
                )
                human_player = GamePlayer(
                    kind="human",
                    model=HUMAN_MODEL_ID,
                    name="Human",
                    temperature=0.0,
                    reasoning=None,
                    color=chess.WHITE if human_color == "white" else chess.BLACK,
                )
                white = human_player if human_color == "white" else opponent_player
                black = opponent_player if human_color == "white" else human_player
            elif mode == "llm_vs_llm":
                white_model = payload.get("whiteModel")
                black_model = payload.get("blackModel")
                if not white_model or not black_model:
                    return jsonify({"error": "Both whiteModel and blackModel are required."}), 400
                white_spec = get_player_spec(white_model, catalog)
                black_spec = get_player_spec(black_model, catalog)
                if white_spec is None or black_spec is None:
                    return jsonify({"error": "One or more models not found."}), 404
                white_model_id, white_temp, white_reasoning, white_kind = white_spec
                black_model_id, black_temp, black_reasoning, black_kind = black_spec
                # Check if any LLM player and no API key
                if (white_kind == "llm" or black_kind == "llm") and not client:
                    return jsonify({"error": "OPENROUTER_API_KEY is not set."}), 400
                white = GamePlayer(
                    kind=white_kind,
                    model=white_model_id,
                    name=white_model_id,
                    temperature=white_temp,
                    reasoning=white_reasoning,
                    color=chess.WHITE,
                )
                black = GamePlayer(
                    kind=black_kind,
                    model=black_model_id,
                    name=black_model_id,
                    temperature=black_temp,
                    reasoning=black_reasoning,
                    color=chess.BLACK,
                )
                opening_choice = payload.get("opening")
                if opening_choice and opening_choice != "random":
                    return jsonify({"error": "Invalid opening selection."}), 400
            else:
                return jsonify({"error": "mode must be human_vs_llm or llm_vs_llm."}), 400

            game = GameState(
                game_id=game_id,
                mode=mode,
                white=white,
                black=black,
                max_plies=max_plies,
                delay=delay,
            )
            game.engine.set_fen_position(chess.STARTING_FEN)
        except BaseException as exc:
            return jsonify({"error": str(exc)}), 500

        with GAMES_LOCK:
            GAMES[game_id] = game

        if mode == "llm_vs_llm":
            if opening_choice == "random" and OPENING_LIBRARY:
                opening = random.choice(OPENING_LIBRARY)
                Thread(
                    target=run_opening_then_loop,
                    args=(game_id, opening, client, payload.get("timeout"), OPENING_DELAY),
                    daemon=True,
                ).start()
            else:
                Thread(target=run_game_loop, args=(game_id, client, payload.get("timeout")), daemon=True).start()
        else:
            # Human vs AI mode - start game loop if it's AI's turn
            if opponent_player and game.board.turn == opponent_player.color:
                Thread(
                    target=run_game_loop,
                    args=(game_id, client, payload.get("timeout")),
                    daemon=True,
                ).start()

        return jsonify(game_snapshot(game))

    @app.route("/api/game/<game_id>")
    def api_game_state(game_id: str) -> object:
        game = get_game(game_id)
        with game.lock:
            snapshot = game_snapshot(game)
        return jsonify(snapshot)

    @app.route("/api/game/<game_id>/stream")
    def api_game_stream(game_id: str) -> object:
        game = get_game(game_id)

        def stream() -> object:
            queue = subscribe_game(game_id)
            try:
                with game.lock:
                    initial = game_snapshot(game)
                yield f"data: {json.dumps(initial)}\n\n"
                while True:
                    try:
                        payload = queue.get(timeout=20)
                    except Empty:
                        yield "event: ping\ndata: {}\n\n"
                        continue
                    yield f"data: {json.dumps(payload)}\n\n"
                    if payload.get("finished") or payload.get("error"):
                        break
            finally:
                unsubscribe_game(game_id, queue)

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/game/<game_id>/move", methods=["POST"])
    def api_game_move(game_id: str) -> object:
        game = get_game(game_id)
        payload = request.get_json(silent=True) or {}
        uci = payload.get("uci", "")
        with game.lock:
            if game.finished or game.error:
                return jsonify(game_snapshot(game)), 200
            human = game.white if game.white.kind == "human" else game.black
            if human.kind != "human":
                return jsonify({"error": "This game does not accept human moves."}), 400
            if game.board.turn != human.color:
                return jsonify({"error": "Not your turn yet."}), 400
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                return jsonify({"error": "Invalid move format."}), 400
            if move not in game.board.legal_moves:
                return jsonify({"error": "Illegal move."}), 400
            record_move(game, move, human.model, None, False, None)
            maybe_finalize_game(game)

        if not game.finished:
            llm_player = game.white if game.white.kind == "llm" else game.black
            Thread(target=run_llm_loop, args=(game_id, client, payload.get("timeout")), daemon=True).start()

        with game.lock:
            snapshot = game_snapshot(game)
        return jsonify(snapshot)

    return app


def main() -> None:
    port = int(os.getenv("LLM_CHESS_PORT", "8000"))
    host = os.getenv("LLM_CHESS_HOST", "127.0.0.1")
    app = create_app()
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
