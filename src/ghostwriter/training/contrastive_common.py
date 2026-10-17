"""Shared frozen-encoder projection and evaluation operations."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..data.contrastive import load_split
from ..models.transformer import LyricsTransformer
from .checkpoint import load_model


def build_projection(embedding_size, dropout):
    return nn.Sequential(
        nn.Linear(embedding_size, embedding_size * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.BatchNorm1d(embedding_size * 2),
        nn.Linear(embedding_size * 2, embedding_size),
    )


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base(run_dir, device):
    run_dir = Path(run_dir).resolve()
    checkpoint = run_dir / "best_transformer.pt"
    tokenizer_path = run_dir / "tokenizer.json"
    model, tokenizer = load_model(checkpoint, tokenizer_path, device)
    if not isinstance(model, LyricsTransformer):
        raise TypeError("Contrastive training requires a Transformer checkpoint.")
    if model.context_length is None:
        raise ValueError("The base checkpoint must record its training context length.")
    if model.context_length < 64:
        raise ValueError("The base Transformer needs a context length of at least 64.")
    model.eval()
    return model, tokenizer, checkpoint, tokenizer_path


def pooled_hidden(model, views, masks):
    hidden = model(views, return_hidden=True)
    valid = masks.unsqueeze(-1).to(hidden.dtype)
    return (hidden * valid).sum(dim=1) / valid.sum(dim=1)


def contrastive_logits(vectors, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    logits = (vectors @ vectors.T) / temperature
    logits.fill_diagonal_(float("-inf"))
    batch_size = vectors.size(0) // 2
    targets = (torch.arange(2 * batch_size, device=vectors.device) + batch_size) % (
        2 * batch_size
    )
    return logits, targets


def evaluate_pairs(model, projection, dataset, batch_size, device, temperature, seed):
    """Score fixed excerpt pairs; the same seed regenerates the same pairs."""
    model.eval()
    projection.eval()
    totals = {
        name: {"loss": 0.0, "correct": 0, "anchors": 0}
        for name in ("pooled", "projected")
    }
    evaluated_songs = 0
    # Sampling uses CPU torch.randint; restore the training RNG after evaluation.
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for first, second, first_mask, second_mask in loader:
            songs = first.size(0)
            if songs < 2:
                continue  # One song supplies no negatives.
            views = torch.cat((first, second)).to(device)
            masks = torch.cat((first_mask, second_mask)).to(device)
            pooled = pooled_hidden(model, views, masks)
            vectors = {
                "pooled": F.normalize(pooled, dim=1),
                "projected": F.normalize(projection(pooled), dim=1),
            }
            evaluated_songs += songs
            for name, values in vectors.items():
                logits, targets = contrastive_logits(values, temperature)
                totals[name]["loss"] += F.cross_entropy(logits, targets).item() * (
                    2 * songs
                )
                totals[name]["correct"] += (
                    (logits.argmax(dim=1) == targets).sum().item()
                )
                totals[name]["anchors"] += 2 * songs
    if not evaluated_songs:
        raise ValueError("Evaluation needs at least two eligible songs in one batch.")
    return {
        "eligible_songs": len(dataset),
        "evaluated_songs": evaluated_songs,
        **{
            name: {
                "loss": values["loss"] / values["anchors"],
                "retrieval_accuracy": values["correct"] / values["anchors"],
                "anchors": values["anchors"],
            }
            for name, values in totals.items()
        },
    }


def save_projection(path, projection, metadata):
    path = Path(path)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {"format_version": 1, **metadata, "state_dict": projection.state_dict()},
        temporary,
    )
    temporary.replace(path)


def load_contrastive(contrastive_dir, device="cpu"):
    contrastive_dir = Path(contrastive_dir).resolve()
    config = json.loads((contrastive_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = contrastive_dir / "best_projection.pt"
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported contrastive checkpoint format.")
    run_dir = Path(config["base_run_dir"])
    model, tokenizer, base_path, tokenizer_path = load_base(run_dir, device)
    if (
        file_hash(base_path) != saved["base_checkpoint_sha256"]
        or file_hash(tokenizer_path) != saved["tokenizer_sha256"]
    ):
        raise ValueError(
            "Base checkpoint or tokenizer changed since contrastive training."
        )
    projection = build_projection(model.embedding.embedding_dim, saved["dropout"]).to(
        device
    )
    projection.load_state_dict(saved["state_dict"])
    projection.eval()
    return model, tokenizer, projection, config, saved


def load_evaluation_split(config, split, tokenizer):
    return load_split(
        Path(config["base_run_dir"]) / "processed",
        split,
        tokenizer,
        max_seq_len=config["context_length"],
        min_view_len=config["min_view_len"],
    )
