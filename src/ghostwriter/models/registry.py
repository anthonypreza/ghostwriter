"""Register architectures here; keep each implementation in its own file."""

import inspect
import json
from pathlib import Path

from .gru import LyricsGRU
from .lstm import LyricsLSTM
from .rnn import LyricsRNN
from .transformer import LyricsTransformer

# Add imports and entries here when implementing the next architectures:
# from .lstm import LyricsLSTM
# MODEL_REGISTRY["lstm"] = LyricsLSTM
MODEL_REGISTRY = {
    "gru": LyricsGRU,
    "rnn": LyricsRNN,
    "lstm": LyricsLSTM,
    "transformer": LyricsTransformer,
}


def build_model(name, vocab_size, config=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Registered models: {', '.join(MODEL_REGISTRY)}"
        )
    config = dict(config or {})
    if "vocab_size" in config:
        raise ValueError(
            "vocab_size is supplied by the tokenizer, not model configuration."
        )
    model_class = MODEL_REGISTRY[name]
    # Persist constructor defaults too, so changing defaults later does not
    # silently change how an existing checkpoint is reconstructed.
    defaults = {
        key: parameter.default
        for key, parameter in inspect.signature(model_class).parameters.items()
        if key != "vocab_size"
        and parameter.default is not inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    model = model_class(vocab_size=vocab_size, **config)
    model.model_config = {**defaults, **config}
    return model


def read_model_config(path):
    if path is None:
        return {}
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(
            "Model configuration must be a JSON object of constructor arguments."
        )
    if "vocab_size" in config:
        raise ValueError(
            "Remove vocab_size from model configuration; it comes from the tokenizer."
        )
    return config


def add_model_arguments(parser):
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="gru")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="JSON file containing model constructor arguments (excluding vocab_size).",
    )
