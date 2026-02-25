"""Training script for the Chess Eval Transformer.

Usage::

    python -m llm_chess.chess_transformer.train \\
        --data data/evals.jsonl \\
        --val-data data/val_games.pgn \\
        --epochs 10 \\
        --batch-size 128 \\
        --lr 3e-4

The script will automatically pick the best available device
(CUDA > MPS > CPU).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from .data_processing import unwarp_eval, _EVAL_CAP
from .dataset import EvalDataset
from .model_architecture import ChessEvalTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _collate(
    batch: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack samples into batched tensors."""
    states, evals = zip(*batch)
    return (
        torch.stack(states),
        torch.tensor(evals, dtype=torch.float32).unsqueeze(1),
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (val_mse, val_mae_cp)."""
    model.eval()
    val_mse_sum = 0.0
    val_mae_cp_sum = 0.0
    val_samples = 0
    with torch.no_grad():
        for states_v, eval_targets_v in val_loader:
            states_v = states_v.to(device)
            eval_targets_v = eval_targets_v.to(device)
            pred_v = model(states_v)
            val_mse_sum += criterion(pred_v, eval_targets_v).item() * states_v.size(0)
            error_cp_v = (pred_v.cpu() - eval_targets_v.cpu()) * _EVAL_CAP * 100.0
            val_mae_cp_sum += error_cp_v.abs().mean().item() * states_v.size(0)
            val_samples += states_v.size(0)
    model.train()
    val_mse = val_mse_sum / max(val_samples, 1)
    val_mae_cp = val_mae_cp_sum / max(val_samples, 1)
    return val_mse, val_mae_cp


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str | Path,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    max_positions: int | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    log_every: int = 50,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    val_data: str | Path | None = None,
    val_every: int = 2000,
) -> None:
    """Run the full training loop.

    Parameters
    ----------
    data_path:      Path to the JSONL eval data.
    epochs:         Number of full passes over the dataset.
    batch_size:     Mini-batch size.
    lr:             Peak learning rate for AdamW.
    weight_decay:   AdamW weight-decay coefficient.
    d_model:        Transformer hidden dimension.
    n_heads:        Number of attention heads.
    n_layers:       Number of Transformer encoder layers.
    max_positions:  Cap on number of positions to load (None = all).
    checkpoint_dir: Directory for saving model checkpoints.
    log_every:      Print progress every N batches.
    wandb_project:  W&B project name. None disables W&B logging.
    wandb_run_name: Optional W&B run name.
    val_data:       Path to validation JSONL data. None disables validation.
    val_every:      Run validation every N batches (0 = end of epoch only).
    """
    device = _select_device()
    logger.info("Using device: %s", device)

    # --- Dataset & loader ---
    dataset = EvalDataset(data_path, max_positions=max_positions)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        pin_memory=(device.type == "cuda"),
    )
    logger.info("Dataset: %d positions, ~%d batches/epoch", len(dataset), len(loader))

    # --- Validation dataset & loader ---
    val_loader = None
    if val_data is not None:
        val_path = Path(val_data)
        if val_path.exists():
            val_dataset = EvalDataset(val_path)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate,
                pin_memory=(device.type == "cuda"),
            )
            logger.info("Validation: %d positions, ~%d batches", len(val_dataset), len(val_loader))
        else:
            logger.warning("Validation data not found: %s — skipping validation", val_path)

    # --- Model ---
    model = ChessEvalTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{param_count:,}")

    # --- Optimiser & loss ---
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # --- W&B ---
    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "data_path": str(data_path),
                "dataset_size": len(dataset),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "param_count": param_count,
                "device": str(device),
                "eval_cap": _EVAL_CAP,
                "val_every": val_every,
            },
        )
        wandb.watch(model, log="gradients", log_freq=log_every)

    # --- Checkpoint directory ---
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_mae = float("inf")

    def _maybe_validate_and_checkpoint() -> None:
        """Run validation if available, save checkpoint if val improved."""
        nonlocal best_val_mae
        if val_loader is None:
            return

        val_mse, val_mae_cp = _run_validation(model, val_loader, criterion, device)
        logger.info(
            "  Val @ step %d — MSE %.6f  MAE(cp) %.1f",
            global_step, val_mse, val_mae_cp,
        )

        if use_wandb:
            wandb.log({
                "val/mse": val_mse,
                "val/mae_cp": val_mae_cp,
            }, step=global_step)

        if val_mae_cp < best_val_mae:
            best_val_mae = val_mae_cp
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_mse": val_mse,
                    "val_mae_cp": val_mae_cp,
                },
                ckpt_path,
            )
            logger.info("  New best val MAE(cp): %.1f — saved %s", best_val_mae, ckpt_path)
        else:
            logger.info("  Val MAE(cp) %.1f did not improve (best: %.1f)", val_mae_cp, best_val_mae)

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_mse = 0.0
        epoch_mae_cp = 0.0
        epoch_samples = 0
        t0 = time.time()

        for batch_idx, (states, eval_targets) in enumerate(loader, 1):
            states = states.to(device)
            eval_targets = eval_targets.to(device)

            # Forward.
            eval_pred = model(states)

            # Loss (MSE on warped eval).
            loss = criterion(eval_pred, eval_targets)

            # Backward.
            optimiser.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            global_step += 1

            # Bookkeeping.
            bs = states.size(0)
            epoch_mse += loss.item() * bs
            epoch_samples += bs

            # MAE in centipawn space (linear unscaling).
            with torch.no_grad():
                pred_raw = eval_pred.detach().cpu()
                tgt_raw = eval_targets.detach().cpu()
                error_cp = (pred_raw - tgt_raw) * _EVAL_CAP * 100.0
                mae_cp = error_cp.abs().mean().item()
                rmse_cp = (error_cp ** 2).mean().sqrt().item()
                epoch_mae_cp += mae_cp * bs

            # Per-batch W&B logging.
            if use_wandb:
                wandb.log({
                    "batch/mse": loss.item(),
                    "batch/mae_cp": mae_cp,
                    "batch/rmse_cp": rmse_cp,
                    "batch/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "batch/lr": optimiser.param_groups[0]["lr"],
                }, step=global_step)

            if batch_idx % log_every == 0:
                logger.info(
                    "  [%d/%d] batch %d/%d — MSE %.6f  MAE(cp) %.1f",
                    epoch, epochs, batch_idx, len(loader),
                    loss.item(), mae_cp,
                )

            # Intra-epoch validation.
            if val_every > 0 and val_loader is not None and batch_idx % val_every == 0:
                _maybe_validate_and_checkpoint()

        # Epoch summary.
        dt = time.time() - t0
        avg_mse = epoch_mse / max(epoch_samples, 1)
        avg_mae_cp = epoch_mae_cp / max(epoch_samples, 1)
        samples_per_sec = epoch_samples / dt if dt > 0 else 0
        logger.info(
            "Epoch %d/%d — MSE %.6f  MAE(cp) %.1f  (%.1f samples/s, %.1fs)",
            epoch, epochs, avg_mse, avg_mae_cp, samples_per_sec, dt,
        )

        if use_wandb:
            wandb.log({
                "epoch/mse": avg_mse,
                "epoch/mae_cp": avg_mae_cp,
                "epoch/duration_s": dt,
                "epoch/samples_per_sec": samples_per_sec,
                "epoch": epoch,
            }, step=global_step)

        # End-of-epoch validation.
        _maybe_validate_and_checkpoint()

    logger.info("Training complete. Best val MAE(cp): %.1f", best_val_mae)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Chess Eval Transformer")
    parser.add_argument("--data", required=True, help="Path to JSONL eval data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--wandb-project", default="chess-eval", help="W&B project name (--no-wandb to disable)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    parser.add_argument("--val-data", default=None, help="Path to validation JSONL data")
    parser.add_argument("--val-every", type=int, default=2000, help="Validate every N batches (0 = epoch end only)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_positions=args.max_positions,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        val_data=args.val_data,
        val_every=args.val_every,
    )


if __name__ == "__main__":
    main()
