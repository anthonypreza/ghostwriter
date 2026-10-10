import math
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import LyricsDataset
from ..data.tokenizer import Tokenizer
from ..models.registry import build_model, add_model_arguments, read_model_config
from .checkpoint import save_model

NUM_EPOCHS = 30
BATCH_SIZE = 32
PATIENCE = 5

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_PATH = PROJECT_ROOT / "best_gru.pt"


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
):
    """Train using only training and validation songs; return learning history."""
    import builtins
    from functools import partial

    print = partial(builtins.print, flush=True)
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
        lr=1e-3,
        weight_decay=1e-2,
    )
    grad_clip_norm = 1.0
    training_settings = {
        "seq_len": seq_len,
        "stride": stride,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "seed": seed,
        "optimizer": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": optimizer.param_groups[0]["weight_decay"],
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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        model_config = read_model_config(args.model_config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    checkpoint = args.checkpoint or PROJECT_ROOT / f"best_{args.model}.pt"
    result = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device_name=args.device,
        seed=args.seed,
        checkpoint_path=checkpoint,
        history_path=checkpoint.with_suffix(".history.json"),
        model_name=args.model,
        model_config=model_config,
    )
    print(f"Best epoch: {result['best_epoch']} | Checkpoint: {checkpoint}")
    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
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
