"""Generate JSONL evaluation data from random-move games + Stockfish evals.

Plays games with random legal moves, then evaluates every non-terminal
position with Stockfish at the specified depth. Outputs one JSONL line
per position.

Usage::

    python -m llm_chess.chess_transformer.generate_games \\
        --output data/evals.jsonl \\
        --games 5000 \\
        --depth 12
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import chess
from stockfish import Stockfish

from llm_chess.cli import stockfish_path

logger = logging.getLogger(__name__)


def _create_engine(depth: int) -> Stockfish:
    """Create a Stockfish instance at the given search depth."""
    engine = Stockfish(path=stockfish_path())
    engine.set_depth(depth)
    return engine


def _play_random_game(max_plies: int = 300) -> list[str]:
    """Play a game of random legal moves, return list of FENs for non-terminal positions."""
    board = chess.Board()
    fens: list[str] = []

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        fens.append(board.fen())
        move = random.choice(list(board.legal_moves))
        board.push(move)

    return fens


def _evaluate_position(engine: Stockfish, fen: str) -> dict | None:
    """Evaluate a single FEN with Stockfish. Returns eval dict or None on error."""
    try:
        engine.set_fen_position(fen)
        evaluation = engine.get_evaluation()
        return {
            "fen": fen,
            "eval_type": evaluation["type"],
            "eval_value": evaluation["value"],
        }
    except Exception as exc:
        logger.warning("Failed to evaluate %s: %s", fen, exc)
        return None


def generate(
    output: str | Path,
    num_games: int = 1000,
    depth: int = 12,
    max_plies: int = 300,
) -> None:
    """Generate evaluation data and write JSONL.

    Parameters
    ----------
    output:     Path to the output JSONL file.
    num_games:  Number of random games to play.
    depth:      Stockfish search depth for evaluation.
    max_plies:  Maximum plies per game.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = _create_engine(depth)

    logger.info(
        "Generating evals from %d random games (depth=%d) -> %s",
        num_games, depth, output_path,
    )

    total_positions = 0
    t0 = time.time()

    with open(output_path, "a", encoding="utf-8") as f:
        for i in range(1, num_games + 1):
            fens = _play_random_game(max_plies=max_plies)

            for fen in fens:
                record = _evaluate_position(engine, fen)
                if record is not None:
                    f.write(json.dumps(record) + "\n")
                    total_positions += 1

            if i % 10 == 0 or i == num_games:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    "  [%d/%d] %.1f games/sec, %d positions so far",
                    i, num_games, rate, total_positions,
                )

    dt = time.time() - t0
    logger.info(
        "Done. %d games -> %d positions in %.1fs (%.1f games/sec)",
        num_games, total_positions, dt, num_games / dt if dt > 0 else 0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSONL evaluation data from random games + Stockfish",
    )
    parser.add_argument(
        "--output", "-o", default="data/evals.jsonl",
        help="Output JSONL file path (default: data/evals.jsonl)",
    )
    parser.add_argument("--games", "-n", type=int, default=1000, help="Number of games")
    parser.add_argument("--depth", "-d", type=int, default=12, help="Stockfish eval depth")
    parser.add_argument("--max-plies", type=int, default=300, help="Max plies per game")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    generate(
        output=args.output,
        num_games=args.games,
        depth=args.depth,
        max_plies=args.max_plies,
    )


if __name__ == "__main__":
    main()
