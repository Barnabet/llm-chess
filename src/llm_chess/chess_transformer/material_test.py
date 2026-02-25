"""Quick sanity check: can the model learn material counting?

Generates random positions, computes material balance as the target,
and trains the same architecture for a few hundred steps. If it can't
even learn this, the architecture is the bottleneck.

Usage::

    python -m llm_chess.chess_transformer.material_test
"""

import logging
import random

import chess
import torch

from .data_processing import board_to_tensor
from .model_architecture import ChessEvalTransformer

logger = logging.getLogger(__name__)

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

CAP = 20.0


def material_balance(board: chess.Board) -> float:
    """Material balance from side-to-move perspective, in pawns."""
    bal = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            bal += val if piece.color == chess.WHITE else -val
    if board.turn == chess.BLACK:
        bal = -bal
    return float(bal)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Generate random positions ---
    random.seed(42)
    positions = []
    for _ in range(2000):
        board = chess.Board()
        for _ in range(random.randint(5, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        if not board.is_game_over():
            positions.append((board_to_tensor(board), material_balance(board)))

    logger.info("Generated %d positions", len(positions))
    logger.info(
        "Material balance range: %d to %d pawns",
        int(min(b for _, b in positions)),
        int(max(b for _, b in positions)),
    )

    states = torch.stack([s for s, _ in positions])
    targets = torch.tensor(
        [b / CAP for _, b in positions], dtype=torch.float32,
    ).unsqueeze(1).clamp(-1, 1)

    # --- Model (same arch as real training) ---
    model = ChessEvalTransformer(d_model=256, n_heads=8, n_layers=6)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{param_count:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.MSELoss()

    # --- Train ---
    model.train()
    batch_size = 128
    n_steps = 300
    for step in range(n_steps):
        idx = torch.randint(0, len(states), (batch_size,))
        pred = model(states[idx])
        loss = criterion(pred, targets[idx])

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        print(f"\rStep {step + 1}/{n_steps} — loss {loss.item():.6f}", end="", flush=True)

        if (step + 1) % 50 == 0:
            with torch.no_grad():
                all_pred = model(states)
                mae = (all_pred - targets).abs().mean().item() * CAP
            print(f"\rStep {step + 1}/{n_steps} — loss {loss.item():.6f}  MAE {mae:.2f} pawns")
    print()

    # --- Final eval ---
    model.eval()
    with torch.no_grad():
        all_pred = model(states)
        errors = (all_pred - targets).abs() * CAP
        logger.info("Final MAE: %.2f pawns (%dcp)", errors.mean(), int(errors.mean() * 100))
        logger.info("Median error: %.2f pawns", errors.median())

        for i in range(5):
            p = all_pred[i].item() * CAP
            a = targets[i].item() * CAP
            logger.info("  Pred: %+.1f  Actual: %+.1f  Error: %.1f", p, a, abs(p - a))


if __name__ == "__main__":
    main()
