import json
import os
import re
from pathlib import Path

RESERVED_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<NEWLINE>",
    "<VERSE>",
    "<CHORUS>",
    "<BRIDGE>",
    "<PRE_CHORUS>",
    "<POST_CHORUS>",
    "<INTRO>",
    "<OUTRO>",
    "<REFRAIN>",
    "<INTERLUDE>",
    "<HOOK>",
]

CURRENT_DIR = os.path.dirname(__file__)
PROCESSED_DIR = Path(f"{CURRENT_DIR}/../../../data/processed")
TRAIN_DIR = PROCESSED_DIR / "train"


def tokenize(text, token_level="word", section_tokens=True, lowercase=True):
    if token_level not in ("word", "char"):
        raise ValueError("token_level must be 'word' or 'char'.")
    reserved_tokens = RESERVED_TOKENS if section_tokens else RESERVED_TOKENS[:5]
    if not section_tokens:
        text = re.sub("|".join(map(re.escape, RESERVED_TOKENS[5:])), "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if token_level == "char":
        # Keep spaces and punctuation; control tokens stay atomic and uppercase.
        text = text.replace("\n", "<NEWLINE>")
        reserved_pattern = "|".join(map(re.escape, reserved_tokens))
        tokens = re.findall(rf"(?:{reserved_pattern})|[\s\S]", text)
        return [
            char
            for token in tokens
            for char in (
                [token]
                if token in reserved_tokens
                else list(token.lower() if lowercase else token)
            )
        ]
    text = text.replace("’", "'")
    text = text.replace("\n", " <NEWLINE> ")

    reserved_pattern = "|".join(
        re.escape(token) for token in sorted(RESERVED_TOKENS, key=len, reverse=True)
    )

    pattern = rf"(?:{reserved_pattern})|\w+(?:['’]\w+)*|[^\w\s]"
    tokens = re.findall(pattern, text)
    return [t.lower() if t not in RESERVED_TOKENS else t for t in tokens]


def build_vocab(
    processed_dir=TRAIN_DIR,
    min_freq=None,
    token_level="word",
    section_tokens=True,
    lowercase=True,
):
    min_freq = min_freq if min_freq is not None else (1 if token_level == "char" else 2)
    if min_freq < 1:
        raise ValueError("min_freq must be positive.")
    reserved_tokens = RESERVED_TOKENS if section_tokens else RESERVED_TOKENS[:5]
    tokens = {}
    for f in sorted(Path(processed_dir).glob("*.txt")):
        with f.open("r", encoding="utf-8") as lyric_file:
            lyric_tokens = tokenize(
                lyric_file.read(), token_level, section_tokens, lowercase
            )
            for t in lyric_tokens:
                if t not in tokens:
                    tokens[t] = 1
                else:
                    tokens[t] += 1
    # remove tokens that aren't frequent
    tokens = [
        (k, v) for k, v in tokens.items() if v >= min_freq and k not in reserved_tokens
    ]
    tokens = [token[0] for token in sorted(tokens, key=lambda t: t[1], reverse=True)]
    all_tokens = reserved_tokens + tokens

    # construct vocabulary mapping
    vocab_map = dict(zip(all_tokens, range(len(all_tokens))))

    return vocab_map


class Tokenizer:
    def __init__(self, token_level="word", section_tokens=True):
        if token_level not in ("word", "char") or not isinstance(section_tokens, bool):
            raise ValueError("Invalid token level or section-token setting.")
        self.token_level = token_level
        self.section_tokens = section_tokens
        self.lowercase = True
        self.vocab_map = {}
        self.id_to_token = {}
        self.pad_token_id = 0

    def build_vocab(self, processed_dir=TRAIN_DIR, min_freq=None):
        self.vocab_map = build_vocab(
            processed_dir,
            min_freq,
            self.token_level,
            self.section_tokens,
            self.lowercase,
        )
        self.id_to_token = {v: k for k, v in self.vocab_map.items()}

    def _require_vocab(self):
        if not self.vocab_map:
            raise ValueError("Build or load a vocabulary before using the tokenizer.")

    def encode(self, text):
        self._require_vocab()
        return [
            self.vocab_map.get(t, self.vocab_map["<UNK>"])
            for t in tokenize(
                text, self.token_level, self.section_tokens, self.lowercase
            )
        ]

    def decode(self, token_ids):
        self._require_vocab()
        return [self.id_to_token[token_id] for token_id in token_ids]

    def save(self, path):
        self._require_vocab()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store tokens in ID order so they retain the same IDs after loading.
        tokens = [self.id_to_token[i] for i in range(len(self.id_to_token))]
        path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "token_level": self.token_level,
                    "section_tokens": self.section_tokens,
                    "lowercase": self.lowercase,
                    "tokens": tokens,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("token_level") == "bpe":
            from .bpe_tokenizer import BPETokenizer

            return BPETokenizer.load(path)
        if not isinstance(data, dict) or data.get("version") not in (1, 2):
            raise ValueError("Unsupported tokenizer file format.")
        tokenizer = (
            cls()
            if data["version"] == 1
            else cls(data.get("token_level"), data.get("section_tokens"))
        )
        reserved_tokens = (
            RESERVED_TOKENS if tokenizer.section_tokens else RESERVED_TOKENS[:5]
        )
        # Earlier character tokenizers preserved case; retain their encoding rules.
        tokenizer.lowercase = data.get("lowercase", tokenizer.token_level == "word")
        if not isinstance(tokenizer.lowercase, bool):
            raise TypeError("Invalid tokenizer lowercase setting.")
        tokens = data.get("tokens")
        if (
            not isinstance(tokens, list)
            or not all(isinstance(token, str) for token in tokens)
            or len(tokens) != len(set(tokens))
            or tokens[: len(reserved_tokens)] != reserved_tokens
        ):
            raise ValueError("Invalid tokenizer vocabulary or reserved tokens.")
        tokenizer.vocab_map = {token: i for i, token in enumerate(tokens)}
        tokenizer.id_to_token = dict(enumerate(tokens))
        return tokenizer


def create_tokenizer(token_level="word", section_tokens=True, vocab_size=None):
    if token_level == "bpe":
        from .bpe_tokenizer import BPETokenizer

        return BPETokenizer(2000 if vocab_size is None else vocab_size, section_tokens)
    if vocab_size is not None:
        raise ValueError("--vocab-size is only supported with --token-level bpe.")
    return Tokenizer(token_level, section_tokens)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a vocabulary from training songs."
    )
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
    parser.add_argument("--min-freq", type=int)
    args = parser.parse_args()
    try:
        tokenizer = create_tokenizer(
            args.token_level, args.section_tokens, args.vocab_size
        )
    except ValueError as error:
        parser.error(str(error))
    tokenizer.build_vocab(min_freq=args.min_freq)
    tokenizer.save(PROCESSED_DIR.parent / "tokenizer.json")
    for f in sorted(TRAIN_DIR.glob("*.txt")):
        with f.open("r", encoding="utf-8") as lyric_file:
            encoded = tokenizer.encode(lyric_file.read())
            print(encoded)
            decoded = tokenizer.decode(encoded)
            print(decoded)


if __name__ == "__main__":
    main()
