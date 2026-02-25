"""PyTorch Dataset that reads positions from a JSONL eval file.

Each sample yields:
    (state_tensor, warped_eval)

where
    state_tensor : [21, 8, 8]  — board observation tensor.
    warped_eval  : float        — Stockfish eval warped to [-1, 1].
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import chess
import torch
from torch.utils.data import Dataset

from .data_processing import board_to_tensor, encode_eval, warp_eval

logger = logging.getLogger(__name__)


class EvalDataset(Dataset):
    """Load chess positions + evaluations from a JSONL file.

    Each line must be a JSON object with keys:
        ``fen``, ``eval_type``, ``eval_value``

    Parameters
    ----------
    jsonl_path:
        Path to the JSONL file.
    max_positions:
        Optional cap on how many positions to load.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        max_positions: int | None = None,
    ) -> None:
        super().__init__()
        # Each entry: (fen, warped_eval)
        self._samples: List[Tuple[str, float]] = []
        self._parse_jsonl(Path(jsonl_path), max_positions)
        logger.info("EvalDataset: %d positions from %s", len(self._samples), jsonl_path)

    def _parse_jsonl(self, path: Path, max_positions: int | None) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                eval_dict = {
                    "type": record["eval_type"],
                    "value": record["eval_value"],
                }
                raw = encode_eval(eval_dict)

                # Stockfish evals are from white's perspective, but the board
                # tensor is oriented from the side-to-move's perspective.
                # Flip the sign for black so eval matches the board orientation.
                fen = record["fen"]
                is_black = " b " in fen
                if is_black:
                    raw = -raw

                warped = warp_eval(raw)

                self._samples.append((fen, warped))

                if max_positions is not None and len(self._samples) >= max_positions:
                    break

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        fen, warped = self._samples[idx]
        board = chess.Board(fen)
        state = board_to_tensor(board)
        return state, warped
