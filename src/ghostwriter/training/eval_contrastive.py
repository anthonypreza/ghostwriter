"""Evaluate saved contrastive embeddings against the frozen Transformer baseline."""

import argparse
import json
from pathlib import Path

import torch

from .contrastive_common import evaluate_pairs, load_contrastive, load_evaluation_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrastive-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, projection, config, saved = load_contrastive(
        args.contrastive_dir, device
    )
    dataset, _ = load_evaluation_split(config, args.split, tokenizer)
    result = evaluate_pairs(
        model,
        projection,
        dataset,
        config["batch_size"],
        device,
        config["temperature"],
        config["seed"],
    )
    print(f"Best projection epoch: {saved['epoch']} | Split: {args.split}")
    print(
        f"Eligible songs: {result['eligible_songs']} | Evaluated: {result['evaluated_songs']}"
    )
    for name in ("pooled", "projected"):
        score = result[name]
        print(
            f"{name.capitalize():9} | Loss {score['loss']:.4f} | "
            f"Same-song retrieval {score['retrieval_accuracy']:.1%} | "
            f"Anchors {score['anchors']}"
        )
    print(
        "Fixed sampled pairs and identical in-batch negatives for both representations."
    )
    print(json.dumps({"split": args.split, **result}, indent=2))


if __name__ == "__main__":
    main()
