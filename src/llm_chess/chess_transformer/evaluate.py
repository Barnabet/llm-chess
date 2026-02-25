"""Evaluate a checkpoint on validation data, broken down by game phase and eval type.

Usage::

    python -m llm_chess.chess_transformer.evaluate \\
        --checkpoint checkpoints/best.pt \\
        --data data/val_games.pgn
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import chess
import torch

from .data_processing import board_to_tensor, encode_eval, warp_eval, unwarp_eval, _EVAL_CAP
from .model_architecture import ChessEvalTransformer

logger = logging.getLogger(__name__)


def _load_model(checkpoint_path: str | Path, device: torch.device) -> ChessEvalTransformer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    d_model = state_dict["input_proj.weight"].shape[0]
    n_layers = sum(
        1 for k in state_dict if k.startswith("encoder.layers.") and k.endswith(".self_attn.in_proj_weight")
    )
    model = ChessEvalTransformer(d_model=d_model, n_heads=8, n_layers=n_layers)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _classify_phase(fen: str) -> str:
    """Classify position as opening/middlegame/endgame based on piece count."""
    board = chess.Board(fen)
    # Count non-pawn, non-king pieces
    minor_major = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is not None and p.piece_type not in (chess.PAWN, chess.KING):
            minor_major += 1
    if minor_major >= 8:
        return "opening"
    elif minor_major >= 4:
        return "middlegame"
    else:
        return "endgame"


def evaluate(
    checkpoint_path: str | Path,
    data_path: str | Path,
    batch_size: int = 128,
) -> None:
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _load_model(checkpoint_path, device)

    # Load all positions
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records)} positions from {data_path}")

    # Precompute metadata
    fens = [r["fen"] for r in records]
    eval_types = [r["eval_type"] for r in records]
    eval_values = [r["eval_value"] for r in records]
    phases = [_classify_phase(fen) for fen in fens]

    # Compute targets (warped) — flip sign for black to match board orientation
    targets_warped = []
    for r in records:
        raw = encode_eval({"type": r["eval_type"], "value": r["eval_value"]})
        if " b " in r["fen"]:
            raw = -raw
        targets_warped.append(warp_eval(raw))

    # Run inference in batches
    preds_warped = []
    with torch.no_grad():
        for i in range(0, len(fens), batch_size):
            batch_fens = fens[i:i + batch_size]
            boards = [chess.Board(fen) for fen in batch_fens]
            tensors = torch.stack([board_to_tensor(b) for b in boards]).to(device)
            out = model(tensors).squeeze(-1).cpu().tolist()
            preds_warped.extend(out)
            if (i // batch_size) % 20 == 0:
                print(f"  Inference: {i + len(batch_fens)}/{len(fens)}", end="\r")

    print()

    # Compute errors in centipawns
    errors_cp = []
    for pred_w, tgt_w in zip(preds_warped, targets_warped):
        errors_cp.append((pred_w - tgt_w) * _EVAL_CAP * 100.0)

    # Build buckets
    buckets = {
        "all": [],
        "cp_only": [],
        "mate_only": [],
        "opening": [],
        "middlegame": [],
        "endgame": [],
        "opening_cp": [],
        "middlegame_cp": [],
        "endgame_cp": [],
    }

    # Eval magnitude buckets (cp only)
    mag_buckets = {
        "|eval| < 100cp": [],
        "100-300cp": [],
        "300-1000cp": [],
        "> 1000cp": [],
    }

    for i, err in enumerate(errors_cp):
        buckets["all"].append(err)
        phase = phases[i]
        buckets[phase].append(err)

        if eval_types[i] == "cp":
            buckets["cp_only"].append(err)
            buckets[f"{phase}_cp"].append(err)

            abs_val = abs(eval_values[i])
            if abs_val < 100:
                mag_buckets["|eval| < 100cp"].append(err)
            elif abs_val < 300:
                mag_buckets["100-300cp"].append(err)
            elif abs_val < 1000:
                mag_buckets["300-1000cp"].append(err)
            else:
                mag_buckets["> 1000cp"].append(err)
        else:
            buckets["mate_only"].append(err)

    # Also compute sign accuracy (does model get the direction right?)
    sign_buckets = {
        "cp_only": {"correct": 0, "total": 0},
        "opening_cp": {"correct": 0, "total": 0},
        "middlegame_cp": {"correct": 0, "total": 0},
        "endgame_cp": {"correct": 0, "total": 0},
    }

    for i in range(len(records)):
        if eval_types[i] != "cp":
            continue
        if eval_values[i] == 0:
            continue  # skip perfectly equal positions
        pred_sign = 1 if preds_warped[i] > 0 else -1
        true_sign = 1 if eval_values[i] > 0 else -1
        phase = phases[i]
        for key in ["cp_only", f"{phase}_cp"]:
            sign_buckets[key]["total"] += 1
            if pred_sign == true_sign:
                sign_buckets[key]["correct"] += 1

    # Print results
    def _stats(errs):
        if not errs:
            return "  (no samples)"
        import statistics
        abs_errs = [abs(e) for e in errs]
        mae = sum(abs_errs) / len(abs_errs)
        rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
        median_ae = statistics.median(abs_errs)
        return f"  N={len(errs):>7d}  MAE={mae:6.1f}cp  RMSE={rmse:6.1f}cp  MedianAE={median_ae:6.1f}cp"

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n--- By type ---")
    for key in ["all", "cp_only", "mate_only"]:
        print(f"  {key:20s} {_stats(buckets[key])}")

    print("\n--- By phase (all positions) ---")
    for key in ["opening", "middlegame", "endgame"]:
        print(f"  {key:20s} {_stats(buckets[key])}")

    print("\n--- By phase (cp only) ---")
    for key in ["opening_cp", "middlegame_cp", "endgame_cp"]:
        print(f"  {key:20s} {_stats(buckets[key])}")

    print("\n--- By eval magnitude (cp only) ---")
    for key, errs in mag_buckets.items():
        print(f"  {key:20s} {_stats(errs)}")

    print("\n--- Sign accuracy (cp only, excl. eval=0) ---")
    for key in ["cp_only", "opening_cp", "middlegame_cp", "endgame_cp"]:
        d = sign_buckets[key]
        if d["total"] > 0:
            pct = 100.0 * d["correct"] / d["total"]
            print(f"  {key:20s}   {d['correct']}/{d['total']} = {pct:.1f}%")

    # Checkpoint metadata
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print("\n--- Checkpoint info ---")
    for key in ["global_step", "epoch", "val_mse", "val_mae_cp", "train_mse", "train_mae_cp"]:
        if key in ckpt:
            print(f"  {key}: {ckpt[key]}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate chess eval checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data", default="data/val_games.pgn")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.batch_size)


if __name__ == "__main__":
    main()
