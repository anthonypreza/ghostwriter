"""Train a projection head on a frozen, pretrained lyric Transformer."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.contrastive import load_split
from .contrastive_common import (
    build_projection,
    contrastive_logits,
    evaluate_pairs,
    file_hash,
    load_base,
    load_evaluation_split,
    pooled_hidden,
    save_projection,
)

BATCH_SIZE = 32
NUM_EPOCHS = 30


def train(
    run_dir,
    output_dir,
    model,
    tokenizer,
    base_checkpoint,
    tokenizer_path,
    device,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    dropout=0.1,
    temperature=0.1,
    learning_rate=5e-4,
    weight_decay=0.01,
    seed=42,
    min_view_len=64,
):
    if epochs < 1 or batch_size < 2 or min_view_len < 1:
        raise ValueError(
            "Epochs and minimum view length must be positive; batch size must be at least 2."
        )
    if not 0 <= dropout < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Dropout must be in [0, 1) and temperature must be positive and finite."
        )
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError(
            "Learning rate must be positive and weight decay non-negative."
        )
    if min_view_len > model.context_length:
        raise ValueError("Minimum view length exceeds the base model's context length.")
    torch.manual_seed(seed)
    train_ds, _ = load_split(
        Path(run_dir) / "processed",
        "train",
        tokenizer,
        max_seq_len=model.context_length,
        min_view_len=min_view_len,
    )
    val_ds, _ = load_split(
        Path(run_dir) / "processed",
        "val",
        tokenizer,
        max_seq_len=model.context_length,
        min_view_len=min_view_len,
    )
    if len(train_ds) < 2 or len(val_ds) < 2:
        raise ValueError(
            "Training and validation each need at least two eligible songs."
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    projection = build_projection(model.embedding.embedding_dim, dropout).to(device)
    optimizer = torch.optim.Adam(
        projection.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "base_run_dir": str(Path(run_dir).resolve()),
        "output_dir": str(output_dir),
        "context_length": model.context_length,
        "embedding_size": model.embedding.embedding_dim,
        "min_view_len": min_view_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "dropout": dropout,
        "temperature": temperature,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "encoder": "frozen_causal_transformer",
        "selection_metric": "validation_projected_in_batch_loss",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    source = {
        "base_checkpoint_sha256": file_hash(base_checkpoint),
        "tokenizer_sha256": file_hash(tokenizer_path),
        "dropout": dropout,
    }
    history = []
    best_loss = float("inf")
    best_epoch = 0
    model.eval()
    print(
        f"Eligible songs: train={len(train_ds)}, val={len(val_ds)} | Device: {device}"
    )
    for epoch in range(1, epochs + 1):
        projection.train()
        loss_sum = 0.0
        anchors = 0
        for first, second, first_mask, second_mask in train_loader:
            songs = first.size(0)
            if songs < 2:
                continue
            views = torch.cat((first, second)).to(device)
            masks = torch.cat((first_mask, second_mask)).to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                pooled = pooled_hidden(model, views, masks)
            projected = F.normalize(projection(pooled), dim=1)
            logits, targets = contrastive_logits(projected, temperature)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * (2 * songs)
            anchors += 2 * songs
        if not anchors:
            raise ValueError("Training produced no batches with negative examples.")
        val = evaluate_pairs(
            model, projection, val_ds, batch_size, device, temperature, seed
        )
        train_loss = loss_sum / anchors
        val_loss = val["projected"]["loss"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_projected_loss": val_loss,
                "val_projected_retrieval_accuracy": val["projected"][
                    "retrieval_accuracy"
                ],
                "val_pooled_loss": val["pooled"]["loss"],
                "val_pooled_retrieval_accuracy": val["pooled"]["retrieval_accuracy"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | Train {train_loss:.4f} | "
            f"Val projected {val_loss:.4f} | Retrieval {val['projected']['retrieval_accuracy']:.1%} "
            f"(pooled {val['pooled']['retrieval_accuracy']:.1%})"
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            save_projection(
                output_dir / "best_projection.pt",
                projection,
                {**source, "epoch": epoch, "val_projected_loss": val_loss},
            )
            print(f"  Saved best projection (epoch {epoch})")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    saved = torch.load(
        output_dir / "best_projection.pt", map_location=device, weights_only=True
    )
    projection.load_state_dict(saved["state_dict"])
    test_ds, _ = load_evaluation_split(config, "test", tokenizer)
    report = {
        "best_epoch": best_epoch,
        "best_val_projected_loss": best_loss,
        "val": evaluate_pairs(
            model, projection, val_ds, batch_size, device, temperature, seed
        ),
        "test": evaluate_pairs(
            model, projection, test_ds, batch_size, device, temperature, seed
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    for split in ("val", "test"):
        scores = report[split]
        print(
            f"{split.upper():4} | Projected retrieval {scores['projected']['retrieval_accuracy']:.1%} "
            f"(pooled {scores['pooled']['retrieval_accuracy']:.1%}) | "
            f"Projected loss {scores['projected']['loss']:.4f} | Songs {scores['evaluated_songs']}"
        )
    print(f"Saved contrastive run: {output_dir}")
    print(
        "Evaluate: python -m ghostwriter.training.eval_contrastive "
        f'--contrastive-dir "{output_dir}" --split test'
    )
    print(
        "Visualize: python -m ghostwriter.cluster_contrastive "
        f'--contrastive-dir "{output_dir}" --split test'
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Existing language-model run."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", choices=("transformer",), default="transformer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-view-len", type=int, default=64)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, base_checkpoint, tokenizer_path = load_base(args.run_dir, device)
    output_dir = args.output_dir or (
        args.run_dir / "contrastive" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    train(
        args.run_dir,
        output_dir,
        model,
        tokenizer,
        base_checkpoint,
        tokenizer_path,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        min_view_len=args.min_view_len,
    )


if __name__ == "__main__":
    main()
