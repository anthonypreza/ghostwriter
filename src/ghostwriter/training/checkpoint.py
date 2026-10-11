"""Load architecture-aware checkpoints, plus legacy weights-only GRU files."""

import json
from pathlib import Path

import torch

from ..data.tokenizer import Tokenizer
from ..models.registry import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "best_gru.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "data" / "tokenizer.json"


def load_model(
    checkpoint_path=DEFAULT_CHECKPOINT, tokenizer_path=DEFAULT_TOKENIZER, device="cpu"
):
    tokenizer = Tokenizer.load(tokenizer_path)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in saved:
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format version.")
        state = saved["model_state_dict"]
        name, config = saved["model_name"], saved["model_config"]
        vocab_size = saved["vocab_size"]
        tokens = [tokenizer.id_to_token[i] for i in range(len(tokenizer.vocab_map))]
        if tokens != saved["vocabulary"]:
            raise ValueError(
                "Checkpoint token IDs do not match the supplied tokenizer."
            )
        tokenizer_config = {
            "token_level": tokenizer.token_level,
            "section_tokens": tokenizer.section_tokens,
            "lowercase": tokenizer.lowercase,
        }
        saved_tokenizer_config = dict(
            saved.get(
                "tokenizer_config", {"token_level": "word", "section_tokens": True}
            )
        )
        saved_tokenizer_config.setdefault(
            "lowercase", saved_tokenizer_config["token_level"] == "word"
        )
        if tokenizer_config != saved_tokenizer_config:
            raise ValueError(
                "Checkpoint tokenizer settings do not match the supplied tokenizer."
            )
        if (
            tokenizer.token_level == "bpe"
            and saved.get("tokenizer_backend") != tokenizer.backend_state
        ):
            raise ValueError(
                "Checkpoint BPE merges or preprocessing do not match the supplied tokenizer."
            )
        context_length = saved["seq_len"]
    else:
        # Old GRU checkpoints contain no architecture or tokenizer metadata.
        state = saved
        vocab_size, embedding_size = state["embedding.weight"].shape
        name = "gru"
        config = {
            "embedding_size": embedding_size,
            "hidden_size": state["gru.weight_hh_l0"].shape[1],
            "dropout": 0.0,
        }
        run_config = Path(checkpoint_path).parent / "config.json"
        context_length = None
        if run_config.is_file():
            context_length = json.loads(run_config.read_text(encoding="utf-8")).get(
                "seq_len"
            )
    if context_length is not None and (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length < 1
    ):
        raise ValueError(
            "Checkpoint training sequence length must be a positive integer."
        )
    if vocab_size != len(tokenizer.vocab_map):
        raise ValueError("Checkpoint vocabulary size does not match the tokenizer.")
    model = build_model(name, vocab_size, config).to(device)
    model.load_state_dict(state)
    model.eval()
    model.context_length = context_length
    return model, tokenizer


def resolve_context_length(model, requested=None):
    """Use the training window and reject accidental context mismatches."""
    trained = getattr(model, "context_length", None)
    if requested is not None and (
        not isinstance(requested, int) or isinstance(requested, bool) or requested < 1
    ):
        raise ValueError("Context length must be a positive integer.")
    if trained is not None:
        if requested is not None and requested != trained:
            raise ValueError(
                f"This model was trained with context length {trained}, but {requested} was requested. "
                "Omit the context override to use the checkpoint's training length."
            )
        return trained
    if requested is None:
        raise ValueError(
            "Training context length is unavailable in this legacy checkpoint or its run config. "
            "Supply the original training length explicitly; it cannot be inferred from the weights."
        )
    return requested


def save_model(model, tokenizer, path, model_name, model_config, seq_len, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "model_name": model_name,
            "model_config": dict(model_config),
            "vocab_size": len(tokenizer.vocab_map),
            "tokenizer_config": {
                "token_level": tokenizer.token_level,
                "section_tokens": tokenizer.section_tokens,
                "lowercase": tokenizer.lowercase,
            },
            "vocabulary": [
                tokenizer.id_to_token[i] for i in range(len(tokenizer.vocab_map))
            ],
            "tokenizer_backend": tokenizer.backend_state
            if tokenizer.token_level == "bpe"
            else None,
            "seq_len": seq_len,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)
