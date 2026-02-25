"""Position encoding and evaluation helpers for the chess eval model.

Converts a ``chess.Board`` into a [21, 8, 8] float tensor and provides
functions to encode/decode Stockfish evaluations into a continuous scale.
"""

from __future__ import annotations

import math
from typing import Dict, List

import chess
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Piece-type order used for channel encoding (1-indexed in python-chess).
_PIECE_TYPES: List[int] = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

# Number of channels per single board snapshot:
#   6 piece types x 2 colours = 12 piece planes
#   + 1 "all own pieces" occupancy + 1 "all opponent pieces" occupancy = 14
_PLANES_PER_SNAPSHOT = 14

# Channel layout:
#   0..13   = current snapshot  (14 planes)
#  14       = colour plane (1 = White to move, 0 = Black)
#  15..18   = castling rights (WK, WQ, BK, BQ)
#  19       = halfmove clock (fifty-move counter), normalised
#  20       = fullmove number, normalised
_TOTAL_CHANNELS = _PLANES_PER_SNAPSHOT + 7  # 21


# ---------------------------------------------------------------------------
# Eval encoding / decoding
# ---------------------------------------------------------------------------

_EVAL_CAP = 20.0  # max eval in pawn units (mates clamp here)


def encode_eval(eval_dict: Dict[str, object]) -> float:
    """Convert a Stockfish eval dict to a pawn-unit scalar clamped to ±_EVAL_CAP.

    Parameters
    ----------
    eval_dict:
        ``{"type": "cp", "value": 125}`` or ``{"type": "mate", "value": 3}``.

    Returns
    -------
    float — eval in pawn units, clamped to [-_EVAL_CAP, _EVAL_CAP].
    Mates are mapped to ±_EVAL_CAP.
    """
    eval_type = eval_dict["type"]
    value = eval_dict["value"]

    if eval_type == "cp":
        pawns = value / 100.0
        return max(-_EVAL_CAP, min(_EVAL_CAP, pawns))
    elif eval_type == "mate":
        sign = 1.0 if value > 0 else -1.0
        return sign * _EVAL_CAP
    else:
        raise ValueError(f"Unknown eval type: {eval_type}")


def warp_eval(raw: float) -> float:
    """Linearly scale a raw eval (from ``encode_eval``) into [-1, 1]."""
    return raw / _EVAL_CAP


def unwarp_eval(warped: float) -> float:
    """Inverse of ``warp_eval`` — map [-1, 1] back to pawn units."""
    return warped * _EVAL_CAP


# ---------------------------------------------------------------------------
# Board -> Tensor
# ---------------------------------------------------------------------------

def _board_snapshot(board: chess.Board) -> torch.Tensor:
    """Return a [14, 8, 8] tensor for a single board position.

    Channels 0-5:  current player's pieces (P, N, B, R, Q, K).
    Channels 6-11: opponent's pieces (P, N, B, R, Q, K).
    Channel 12:    all own pieces occupancy.
    Channel 13:    all opponent pieces occupancy.

    The board is always oriented from the current player's perspective
    (rank 0 = the player's back rank).
    """
    planes = torch.zeros(14, 8, 8)
    turn = board.turn  # True = White

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue

        rank = chess.square_rank(sq)
        file = chess.square_file(sq)

        # Flip board so current player is always at the bottom.
        if not turn:
            rank = 7 - rank
            file = 7 - file

        is_own = piece.color == turn
        piece_offset = 0 if is_own else 6
        type_idx = _PIECE_TYPES.index(piece.piece_type)

        planes[piece_offset + type_idx, rank, file] = 1.0

        # Aggregate occupancy planes.
        if is_own:
            planes[12, rank, file] = 1.0
        else:
            planes[13, rank, file] = 1.0

    return planes


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert a board to a [21, 8, 8] tensor.

    Parameters
    ----------
    board:
        The current position.

    Returns
    -------
    torch.Tensor of shape [21, 8, 8].
    """
    planes = torch.zeros(_TOTAL_CHANNELS, 8, 8)

    # --- Snapshot (channels 0..13) ---
    planes[0:14] = _board_snapshot(board)

    # --- Meta planes (channels 14..20) ---
    # 14: colour (1 = White to move).
    if board.turn == chess.WHITE:
        planes[14] = 1.0

    # 15-18: castling rights.
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[15] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[16] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[17] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[18] = 1.0

    # 19: fifty-move clock normalised to [0, 1].
    planes[19] = board.halfmove_clock / 100.0

    # 20: fullmove number normalised (cap at 200 for stability).
    planes[20] = min(board.fullmove_number, 200) / 200.0

    return planes
