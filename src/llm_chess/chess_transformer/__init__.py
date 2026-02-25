"""Chess Eval Transformer — position evaluation via Transformer.

A modular PyTorch codebase for training a Transformer-based chess model
that predicts Stockfish evaluations from board positions.
"""

from .data_processing import board_to_tensor, encode_eval, unwarp_eval, warp_eval, _EVAL_CAP
from .dataset import EvalDataset
from .model_architecture import ChessEvalTransformer

__all__ = [
    "board_to_tensor",
    "encode_eval",
    "warp_eval",
    "unwarp_eval",
    "ChessEvalTransformer",
    "EvalDataset",
]
