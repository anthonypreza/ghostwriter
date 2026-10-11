"""Prepare songs, fit a tokenizer, train a selected model, and evaluate it."""

import argparse
import json
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .console import use_color
from .data.dataset import SEQUENCE_LEN, STRIDE, LyricsDataset
from .data.prepare import prepare_corpus
from .data.tokenizer import create_tokenizer
from .models.registry import add_model_arguments, read_model_settings
from .training.checkpoint import PROJECT_ROOT, load_model
from .training.eval import evaluate
from .training.train import (
    add_optimizer_arguments,
    train,
    validate_optimizer_settings,
    validate_scheduler_settings,
)


class Tee:
    """Send live progress to both the terminal and a saved log."""

    def __init__(self, terminal, log, color=False):
        self.terminal, self.log = terminal, log
        self.color = color

    def style(self, text):
        message = text.strip()
        if message.startswith("FAILED:"):
            color = "1;31"  # Red
        elif message.startswith(
            ("Pipeline interrupted", "No improvement", "Early stopping")
        ):
            color = "33"  # Yellow
        elif message.startswith(("Saved best", "Completed", "Saved checkpoint")):
            color = "1;32"  # Green
        elif re.match(r"\[\d/\d\]", message):
            color = "1;36"  # Cyan stage headings
        elif message.startswith(("Epoch", "VAL ", "TEST ")):
            color = "35"  # Magenta metrics
        elif message.startswith(
            ("Ghostwriter", "Songs:", "Vocabulary:", "Device:", "Windows:")
        ):
            color = "1;34"  # Blue summaries
        else:
            return text
        return f"\033[{color}m{text}\033[0m"

    def write(self, text):
        self.terminal.write(self.style(text) if self.color else text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_pipeline(args, run_dir):
    started = time.perf_counter()
    processed = run_dir / "processed"
    tokenizer_path = run_dir / "tokenizer.json"
    checkpoint = run_dir / f"best_{args.model}.pt"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = {**vars(args), "run_dir": str(run_dir), "device": device, "split_seed": 42}
    (run_dir / "config.json").write_text(
        json.dumps(config, default=str, indent=2), encoding="utf-8"
    )
    print(f"Ghostwriter pipeline | Run: {run_dir}\nRaw songs: {args.raw_dir}")

    print("\n[1/4] Prepare songs and create 80/10/10 splits (seed 42)")
    manifest = prepare_corpus(
        args.raw_dir, processed, section_tokens=args.section_tokens
    )
    print(
        "Songs: "
        + " | ".join(
            f"{split}={len(manifest[split]):,}" for split in ("train", "val", "test")
        )
    )

    print("\n[2/4] Fit and save tokenizer using training songs only")
    tokenizer = create_tokenizer(args.token_level, args.section_tokens, args.vocab_size)
    tokenizer.build_vocab(processed / "train", min_freq=args.min_freq)
    tokenizer.save(tokenizer_path)
    print(
        f"Vocabulary: {len(tokenizer.vocab_map):,} tokens | Minimum frequency: {args.min_freq} | "
        f"Token level: {args.token_level} | Section tokens: {args.section_tokens}"
    )

    print(
        f"\n[3/4] Train {args.model.upper()}; choose best checkpoint using validation loss"
    )
    result = train(
        processed,
        tokenizer_path,
        checkpoint,
        args.epochs,
        args.batch_size,
        args.patience,
        device,
        args.seed,
        args.seq_len,
        args.stride,
        run_dir / "history.json",
        model_name=args.model,
        model_config=args.model_settings,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    config["model_settings"] = result["model_config"]
    config["training_settings"] = result["training_settings"]
    (run_dir / "config.json").write_text(
        json.dumps(config, default=str, indent=2), encoding="utf-8"
    )

    print("\n[4/4] Evaluate best checkpoint with non-overlapping windows")
    model, tokenizer = load_model(checkpoint, tokenizer_path, device)
    metrics = {}
    for split in ("val", "test"):
        dataset = LyricsDataset.from_dir(
            processed / split,
            tokenizer,
            seq_len=args.seq_len,
            stride=args.seq_len,
            pad_token_id=tokenizer.pad_token_id,
        )
        metrics[split] = evaluate(
            model,
            DataLoader(dataset, batch_size=args.batch_size),
            tokenizer.pad_token_id,
            device,
        )
        measured = metrics[split]
        print(
            f"{split.upper():4} | Loss {measured['loss']:.4f} | PPL {measured['perplexity']:.2f} | "
            f"Accuracy {measured['accuracy']:.2%} | Targets {measured['target_tokens']:,}"
        )
    report = {
        "model": args.model,
        "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"],
        "selection_val_loss": result["best_val_loss"],
        "evaluation": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"\nCompleted in {report['elapsed_seconds']:.1f}s | Best epoch: {result['best_epoch']}"
    )
    print(
        f"Saved checkpoint, tokenizer, splits, config, history, metrics, and pipeline.log in:\n{run_dir}"
    )
    print(
        "Generate with:\npython -m ghostwriter.inference.generate "
        f'--checkpoint "{checkpoint}" --tokenizer "{tokenizer_path}" '
        + ('--prompt "<INTRO>" ' if tokenizer.section_tokens else "")
        + f"--max-new-tokens {1000 if tokenizer.token_level == 'char' else 200} --temperature 1.0 --top-k 40"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    add_optimizer_arguments(parser)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="New output folder; defaults to runs/<timestamp>.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--token-level", choices=("word", "char", "bpe"), default="word"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="BPE target vocabulary size; default 2000.",
    )
    parser.add_argument(
        "--section-tokens", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=None,
        help="Default: 2 for words/BPE, 1 for characters; BPE uses this for merges.",
    )
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed; the song split seed stays 42.",
    )
    parser.add_argument(
        "--device", default=None, help="Default: CUDA if available, otherwise CPU."
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Terminal colors; saved logs always remain plain text.",
    )
    args = parser.parse_args()
    if args.min_freq is None:
        args.min_freq = 1 if args.token_level == "char" else 2
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
        args.model_settings = read_model_settings(args)
        create_tokenizer(args.token_level, args.section_tokens, args.vocab_size)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if (
        min(
            args.epochs,
            args.batch_size,
            args.patience,
            args.min_freq,
            args.seq_len,
            args.stride,
        )
        < 1
    ):
        parser.error("All count and window options must be positive.")
    if args.stride > args.seq_len:
        parser.error("--stride must not exceed --seq-len (would skip targets).")
    if not args.raw_dir.is_dir() or len(list(args.raw_dir.glob("*.txt"))) < 6:
        parser.error("--raw-dir must contain at least 6 .txt song files.")
    run_dir = (
        args.run_dir
        or PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    ).resolve()
    if run_dir.exists():
        parser.error(f"Run directory already exists; choose a new directory: {run_dir}")
    run_dir.mkdir(parents=True)
    with (run_dir / "pipeline.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(
            Tee(sys.stdout, log, color=use_color(args.color, sys.stdout))
        ):
            try:
                run_pipeline(args, run_dir)
            except KeyboardInterrupt:
                print(
                    "\nPipeline interrupted. Completed artifacts remain in the run directory."
                )
                raise SystemExit(130)
            except Exception as error:
                print(f"\nFAILED: {type(error).__name__}: {error}")
                print(f"Pipeline stopped; inspect {run_dir / 'pipeline.log'}")
                raise


if __name__ == "__main__":
    main()
