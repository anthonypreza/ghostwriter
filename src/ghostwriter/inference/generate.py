"""Autoregressive generation from a stored language model and its tokenizer."""

import argparse
import re
import sys
import secrets
from pathlib import Path

import torch

from ..console import use_color
from ..training.checkpoint import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER, load_model, resolve_context_length


def render_tokens(tokens):
    text = " ".join(t for t in tokens if t not in ("<PAD>", "<BOS>", "<EOS>"))
    text = re.sub(r" *<NEWLINE> *", "\n", text)
    text = re.sub(r" +([,.!?;:)])", r"\1", text)
    return re.sub(r"\( +", "(", text).strip()


def colorize_lyrics(text):
    """Decorate section markers for display; generated text stays plain."""
    colors = {
        "VERSE": "1;34",  # Blue
        "CHORUS": "1;35",  # Magenta
        "PRE_CHORUS": "1;36",  # Cyan
        "POST_CHORUS": "1;36",
        "BRIDGE": "1;33",  # Yellow
        "INTRO": "1;32",  # Green
        "OUTRO": "1;32",
    }
    return re.sub(
        r"<([A-Z_]+)>",
        lambda match: (
            f"\033[{colors.get(match.group(1), '1;36')}m{match.group(0)}\033[0m"
        ),
        text,
    )


def format_generation(text, args, model, colored=False):
    """Separate one terminal sample from commands and previous samples."""

    def style(value, code):
        return f"\033[{code}m{value}\033[0m" if colored else value

    rule = "=" * 64
    model_name = type(model).__name__.removeprefix("Lyrics")
    context = resolve_context_length(model, args.context_length)
    lines = [
        "",
        style(rule, "36"),
        style("  GHOSTWRITER | GENERATED LYRICS", "1;36"),
        f"  Model: {model_name} | Run: {Path(args.checkpoint).parent.name}",
        f"  Temperature: {args.temperature:g} | Top-k: {args.top_k} | Training context: {context}",
        f"  Seed: {args.seed} | New-token limit: {args.max_new_tokens}",
        style("-" * 64, "36"),
        "",
    ]
    # Indent the lyric body so it stands apart from shell commands.
    displayed = colorize_lyrics(text) if colored else text
    lines.extend("  " + line if line else "" for line in displayed.splitlines())
    if not text:
        lines.append("  (No visible text generated.)")
    lines.extend(["", style("  END OF GENERATION", "1;32"), style(rule, "36"), ""])
    return "\n".join(lines)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt="<INTRO>",
    max_new_tokens=200,
    temperature=1.1,
    top_k=20,
    context_length=None,
    seed=None,
):
    context_length = resolve_context_length(model, context_length)
    if max_new_tokens < 0 or temperature < 0 or top_k < 0 or context_length < 1:
        raise ValueError(
            "Token limit, temperature and top-k must be nonnegative; context must be positive."
        )
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device)
    if seed is None:
        rng.seed()  # Fresh random seed for each generation call.
    else:
        rng.manual_seed(seed)
    ids = [tokenizer.vocab_map["<BOS>"], *tokenizer.encode(prompt)]
    for _ in range(max_new_tokens):
        inputs = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1].clone()  # Last position predicts the next token.
        for token in ("<PAD>", "<BOS>", "<UNK>"):
            logits[tokenizer.vocab_map[token]] = -torch.inf
        if temperature == 0:
            next_id = logits.argmax().item()
        else:
            logits /= temperature
            if top_k:
                values, indices = torch.topk(logits, min(top_k, logits.numel()))
                filtered = torch.full_like(logits, -torch.inf)
                filtered[indices] = values
                logits = filtered
            next_id = torch.multinomial(logits.softmax(dim=-1), 1, generator=rng).item()
        ids.append(next_id)
        if next_id == tokenizer.vocab_map["<EOS>"]:
            break
    return render_tokens(tokenizer.decode(ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", default="<INTRO>")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.1,
        help="0 selects the most likely token; positive values sample.",
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="0 samples from the full vocabulary."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Normally omit: loaded from training metadata. If supplied, must match the training length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible sampling; omitted uses fresh randomness.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Color section markers; automatic mode keeps redirected output plain.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print only plain lyrics, without a terminal frame or colors.",
    )
    args = parser.parse_args()
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, args.device)
    try:
        args.context_length = resolve_context_length(model, args.context_length)
    except ValueError as error:
        parser.error(str(error))
    if args.seed is None:
        args.seed = secrets.randbits(63)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.context_length,
        args.seed,
    )
    colored = not args.plain and use_color(args.color, sys.stdout)
    if sys.stdout.isatty() and not args.plain:
        print(format_generation(text, args, model, colored))
    else:
        print(colorize_lyrics(text) if colored else text)


if __name__ == "__main__":
    main()
