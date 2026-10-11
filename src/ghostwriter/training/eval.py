"""Evaluate a saved language model without starting training."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import LyricsDataset
from .checkpoint import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKENIZER,
    PROJECT_ROOT,
    load_model,
    resolve_context_length,
)


@torch.inference_mode()
def evaluate(model, loader, pad_token_id, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")
    loss_sum, correct, token_count = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)  # (batch, sequence, vocabulary)
        mask = targets != pad_token_id
        loss_sum += criterion(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        ).item()
        correct += ((logits.argmax(dim=-1) == targets) & mask).sum().item()
        token_count += mask.sum().item()
    if token_count == 0:
        raise ValueError("Dataset contains no non-padding target tokens.")
    loss = loss_sum / token_count
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 709 else float("inf"),
        "accuracy": correct / token_count,
        "target_tokens": token_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--data-dir", type=str, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Normally omit: loaded from training metadata. If supplied, must match training.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.seq_len is not None and args.seq_len < 1) or args.batch_size < 1:
        parser.error("Sequence length and batch size must be positive.")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, args.device)
    try:
        seq_len = resolve_context_length(model, args.seq_len)
    except ValueError as error:
        parser.error(str(error))
    dataset = LyricsDataset.from_dir(
        Path(args.data_dir) / args.split,
        tokenizer,
        seq_len=seq_len,
        stride=seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Non-overlapping windows count each next-token target once per song.
    metrics = evaluate(
        model,
        DataLoader(dataset, batch_size=args.batch_size),
        tokenizer.pad_token_id,
        args.device,
    )
    print(
        json.dumps(
            {"split": args.split, "songs": len(dataset.encoded_songs), **metrics},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
