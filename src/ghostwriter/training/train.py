import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import LyricsDataset
from ..data.tokenizer import Tokenizer
from ..models.registry import add_model_arguments, build_model, read_model_settings
from .checkpoint import save_model

NUM_EPOCHS = 30
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_PATH = PROJECT_ROOT / "best_gru.pt"


def add_optimizer_arguments(parser):
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="AdamW learning rate; default 0.0003.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="AdamW weight decay; default 0.01. Use 0 to disable.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("plateau", "none"),
        default="plateau",
        help="Reduce learning rate on stalled validation loss; default plateau.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help="Plateau tolerance; reduce after this many bad epochs plus one (default 2).",
    )
    parser.add_argument("--min-lr", type=float, default=1e-6)


def validate_optimizer_settings(learning_rate, weight_decay):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and greater than zero.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative.")


def validate_scheduler_settings(
    lr_scheduler, lr_factor, lr_patience, min_lr, learning_rate, patience
):
    if lr_scheduler not in ("plateau", "none"):
        raise ValueError("--lr-scheduler must be plateau or none.")
    if not math.isfinite(lr_factor) or not 0 < lr_factor < 1:
        raise ValueError("--lr-factor must be finite and between 0 and 1.")
    if not isinstance(lr_patience, int) or lr_patience < 0:
        raise ValueError("--lr-patience must be a non-negative integer.")
    if not math.isfinite(min_lr) or not 0 <= min_lr <= learning_rate:
        raise ValueError(
            "--min-lr must be finite, non-negative, and no greater than --learning-rate."
        )
    if lr_scheduler == "plateau" and lr_patience >= patience:
        raise ValueError("--lr-patience must be less than early-stopping --patience.")


def train(
    processed_dir=PROCESSED_DIR,
    tokenizer_path=PROCESSED_DIR.parent / "tokenizer.json",
    checkpoint_path=CHECKPOINT_PATH,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
    device_name=None,
    seed=42,
    seq_len=32,
    stride=16,
    history_path=None,
    model_name="gru",
    model_config=None,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler="plateau",
    lr_factor=0.5,
    lr_patience=2,
    min_lr=1e-6,
):
    """Train using only training and validation songs; return learning history."""
    import builtins
    from functools import partial

    print = partial(builtins.print, flush=True)
    validate_optimizer_settings(learning_rate, weight_decay)
    validate_scheduler_settings(
        lr_scheduler, lr_factor, lr_patience, min_lr, learning_rate, patience
    )
    if min(epochs, batch_size, patience, seq_len, stride) < 1:
        raise ValueError(
            "Epochs, batch size, patience, sequence length and stride must be positive."
        )
    torch.manual_seed(seed)
    PROCESSED_DIR = Path(processed_dir)
    CHECKPOINT_PATH = Path(checkpoint_path)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NUM_EPOCHS, BATCH_SIZE, PATIENCE = epochs, batch_size, patience

    tokenizer = Tokenizer.load(path=tokenizer_path)

    train_ds = LyricsDataset.from_dir(
        PROCESSED_DIR / "train",
        tokenizer,
        seq_len=seq_len,
        stride=stride,
        pad_token_id=tokenizer.pad_token_id,
    )
    val_ds = LyricsDataset.from_dir(
        PROCESSED_DIR / "val",
        tokenizer,
        seq_len=seq_len,
        stride=stride,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Model setup
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model_config = dict(model_config or {})
    model = build_model(model_name, len(tokenizer.id_to_token), model_config).to(device)
    model_config = model.model_config

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=0.0,
            threshold_mode="abs",
            min_lr=min_lr,
        )
        if lr_scheduler == "plateau"
        else None
    )
    grad_clip_norm = 1.0
    training_settings = {
        "token_level": tokenizer.token_level,
        "lowercase": tokenizer.lowercase,
        "section_tokens": tokenizer.section_tokens,
        "seq_len": seq_len,
        "stride": stride,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "seed": seed,
        "optimizer": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "lr_scheduler": lr_scheduler,
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
        "min_lr": min_lr,
        "reset_patience_on_lr_reduction": scheduler is not None,
        "grad_clip_norm": grad_clip_norm,
        "vocab_size": len(tokenizer.id_to_token),
    }

    print(
        f"Device: {device} | Model: {model_name} | Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"Model settings: {json.dumps(model_config) if model_config else 'constructor defaults'}"
    )
    print(f"Training settings: {json.dumps(training_settings)}")
    print(
        f"Windows: train={len(train_ds):,}, val={len(val_ds):,} | Batch size: {BATCH_SIZE}"
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
        "next_learning_rate": [],
    }

    def run_epoch(loader, training=True):
        model.train(training)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        last_progress = time.perf_counter()
        for batch_index, (X, y) in enumerate(loader, 1):
            X = X.to(device)
            y = y.to(device)

            # Number of non-padding target tokens
            mask = y != tokenizer.pad_token_id
            valid_tokens = mask.sum().item()

            if valid_tokens == 0:
                continue

            if training:
                optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                # (batch, seq_len, vocab_size)
                predictions = model(X)

                # CrossEntropyLoss expects (N, vocab_size)
                # and targets of shape (N,)
                loss = criterion(
                    predictions.reshape(-1, predictions.size(-1)),
                    y.reshape(-1),
                )

                if training:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )

                    optimizer.step()

            # Token-level accuracy
            with torch.no_grad():
                predicted_tokens = predictions.argmax(dim=-1)

                correct = ((predicted_tokens == y) & mask).sum().item()

            # Weighted by non-padding tokens
            total_loss += loss.item() * valid_tokens
            total_correct += correct
            total_tokens += valid_tokens
            if time.perf_counter() - last_progress >= 30:
                print(
                    f"  {'Train' if training else 'Val'} batch {batch_index}/{len(loader)} | "
                    f"Running loss: {total_loss / total_tokens:.4f}"
                )
                last_progress = time.perf_counter()

        if total_tokens == 0:
            raise ValueError("Dataset contains no valid target tokens")

        return {
            "loss": total_loss / total_tokens,
            "acc": total_correct / total_tokens,
        }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        epoch_started = time.perf_counter()
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            train_loader,
            training=True,
        )

        val_metrics = run_epoch(
            val_loader,
            training=False,
        )

        # Record metrics before early stopping
        for metric, value in train_metrics.items():
            history[f"train_{metric}"].append(value)

        for metric, value in val_metrics.items():
            history[f"val_{metric}"].append(value)

        if not all(math.isfinite(m["loss"]) for m in (train_metrics, val_metrics)):
            raise ValueError("Non-finite training or validation loss; stopping.")
        val_perplexity = (
            math.exp(val_metrics["loss"]) if val_metrics["loss"] < 709 else float("inf")
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.2%} | "
            f"Val Acc: {val_metrics['acc']:.2%} | "
            f"Val PPL: {val_perplexity:.2f} | "
            f"LR: {epoch_lr:.3g} | "
            f"Time: {time.perf_counter() - epoch_started:.1f}s"
        )

        # Save best model based on validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            save_model(
                model,
                tokenizer,
                CHECKPOINT_PATH,
                model_name,
                model_config,
                seq_len,
                best_epoch,
            )
            print(f"  Saved best checkpoint (epoch {best_epoch})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement: {epochs_without_improvement}/{PATIENCE}")

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])
        next_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(epoch_lr)
        history["next_learning_rate"].append(next_lr)
        if next_lr < epoch_lr:
            # Allow a full early-stopping window at the new learning rate.
            epochs_without_improvement = 0
            print(
                f"  Learning rate reduced: {epoch_lr:.3g} -> {next_lr:.3g} | "
                "Early-stopping patience reset"
            )

        if history_path:
            Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            Path(history_path).write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch}")
            break

    return {
        "model": model_name,
        "model_config": model_config,
        "training_settings": training_settings,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history["train_loss"]),
        "history": history,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train a registered language model.")
    add_model_arguments(parser)
    add_optimizer_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        validate_optimizer_settings(args.learning_rate, args.weight_decay)
        validate_scheduler_settings(
            args.lr_scheduler,
            args.lr_factor,
            args.lr_patience,
            args.min_lr,
            args.learning_rate,
            args.patience,
        )
        model_config = read_model_settings(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    checkpoint = args.checkpoint or PROJECT_ROOT / f"best_{args.model}.pt"
    result = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device_name=args.device,
        seed=args.seed,
        seq_len=args.seq_len,
        stride=args.stride,
        checkpoint_path=checkpoint,
        history_path=checkpoint.with_suffix(".history.json"),
        model_name=args.model,
        model_config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    print(f"Best epoch: {result['best_epoch']} | Checkpoint: {checkpoint}")
    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        _, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric in zip(axes, ("loss", "acc")):
            for split in ("train", "val"):
                ax.plot(
                    range(1, result["epochs_completed"] + 1),
                    result["history"][f"{split}_{metric}"],
                    label=split,
                )
            ax.set(xlabel="Epoch", title=metric)
            ax.legend()
            if metric == "acc":
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
