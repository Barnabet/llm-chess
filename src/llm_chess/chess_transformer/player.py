"""Transformer-based chess player using 1-ply minimax.

Evaluates every legal move by making it on a copy of the board,
then scoring the resulting position with the eval model. Picks the
move that leaves the opponent with the worst position (lowest eval
from their perspective).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chess
import torch

from .data_processing import board_to_tensor
from .model_architecture import ChessEvalTransformer

logger = logging.getLogger(__name__)


def _find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """Return the most recently modified .pt file in the directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        return None
    pts = sorted(ckpt_dir.glob("chess_eval_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None


class TransformerPlayer:
    """Chess player backed by a ChessEvalTransformer checkpoint.

    Parameters
    ----------
    checkpoint:
        Path to a ``.pt`` checkpoint file. If None, searches
        ``checkpoint_dir`` for the latest one.
    checkpoint_dir:
        Directory to search for checkpoints (default: ``checkpoints``).
    device:
        Torch device. Auto-detected if None.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Resolve checkpoint path.
        if checkpoint is not None:
            ckpt_path = Path(checkpoint)
        else:
            ckpt_path = _find_latest_checkpoint(checkpoint_dir)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found (checkpoint={checkpoint}, dir={checkpoint_dir})"
            )

        logger.info("Loading transformer checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        state_dict = ckpt["model_state_dict"]

        # Infer d_model from the checkpoint.
        d_model = state_dict["input_proj.weight"].shape[0]
        # Count encoder layers.
        n_layers = sum(
            1 for k in state_dict if k.startswith("encoder.layers.") and k.endswith(".self_attn.in_proj_weight")
        )
        # Infer n_heads from attention projection shape.
        n_heads = 8  # default
        for k, v in state_dict.items():
            if "self_attn.in_proj_weight" in k:
                # in_proj_weight is [3 * d_model, d_model]; head_dim = d_model / n_heads
                # We can't uniquely recover n_heads from this alone, so try common values.
                for candidate in [1, 2, 4, 8, 16, 32]:
                    if d_model % candidate == 0:
                        n_heads = candidate
                break

        self.model = ChessEvalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Transformer player ready: d_model=%d, n_layers=%d, device=%s",
            d_model, n_layers, self.device,
        )

    @torch.no_grad()
    def evaluate_positions(self, boards: list[chess.Board]) -> list[float]:
        """Batch-evaluate a list of board positions.

        Returns warped eval scores in [-1, 1] (positive = good for side to move).
        """
        if not boards:
            return []
        tensors = torch.stack([board_to_tensor(b) for b in boards])
        tensors = tensors.to(self.device)
        evals = self.model(tensors).squeeze(-1).cpu().tolist()
        return evals

    def pick_move(self, board: chess.Board) -> chess.Move:
        """Select the best move via 1-ply minimax.

        For each legal move, make the move on a copy and evaluate the
        resulting position from the opponent's perspective. Pick the
        move that minimises the opponent's eval (i.e. maximises our
        advantage).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Generate all successor positions.
        successor_boards: list[chess.Board] = []
        for move in legal_moves:
            child = board.copy()
            child.push(move)
            successor_boards.append(child)

        # Evaluate from the opponent's perspective (it's their turn after our move).
        # The model outputs eval from the side-to-move's POV.
        # Higher eval = better for the opponent = worse for us.
        # So we pick the move with the LOWEST opponent eval.
        opponent_evals = self.evaluate_positions(successor_boards)

        best_idx = 0
        best_eval = opponent_evals[0]
        for i, ev in enumerate(opponent_evals):
            if ev < best_eval:
                best_eval = ev
                best_idx = i

        chosen = legal_moves[best_idx]
        logger.debug(
            "Transformer chose %s (opponent eval %.4f) from %d legal moves",
            chosen.uci(), best_eval, len(legal_moves),
        )
        return chosen


# Module-level singleton for reuse across games.
_player_instance: Optional[TransformerPlayer] = None


def get_player(
    checkpoint: str | Path | None = None,
    checkpoint_dir: str | Path = "checkpoints",
) -> TransformerPlayer:
    """Get or create the singleton TransformerPlayer."""
    global _player_instance
    if _player_instance is None:
        _player_instance = TransformerPlayer(
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
        )
    return _player_instance
