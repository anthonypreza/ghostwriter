import re
import os
import json
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


def tokenize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'")
    text = text.replace("\n", " <NEWLINE> ")

    reserved_pattern = "|".join(
        re.escape(token) for token in sorted(RESERVED_TOKENS, key=len, reverse=True)
    )

    pattern = rf"(?:{reserved_pattern})|\w+(?:['’]\w+)*|[^\w\s]"
    tokens = re.findall(pattern, text)
    return [t.lower() if t not in RESERVED_TOKENS else t for t in tokens]


def build_vocab(processed_dir=TRAIN_DIR, min_freq=2):
    tokens = {}
    for f in sorted(Path(processed_dir).glob("*.txt")):
        with f.open("r", encoding="utf-8") as lyric_file:
            lyric_tokens = tokenize(lyric_file.read())
            for t in lyric_tokens:
                if t not in tokens:
                    tokens[t] = 1
                else:
                    tokens[t] += 1
    # remove tokens that aren't frequent
    tokens = [
        (k, v) for k, v in tokens.items() if v >= min_freq and k not in RESERVED_TOKENS
    ]
    tokens = [token[0] for token in sorted(tokens, key=lambda t: t[1], reverse=True)]
    all_tokens = RESERVED_TOKENS + tokens

    # construct vocabulary mapping
    vocab_map = dict(zip(all_tokens, range(len(all_tokens))))

    return vocab_map


class Tokenizer:
    def __init__(self):
        self.vocab_map = {}
        self.id_to_token = {}
        self.pad_token_id = 0

    def build_vocab(self, processed_dir=TRAIN_DIR, min_freq=2):
        self.vocab_map = build_vocab(processed_dir, min_freq)
        self.id_to_token = {v: k for k, v in self.vocab_map.items()}

    def _require_vocab(self):
        if not self.vocab_map:
            raise ValueError("Build or load a vocabulary before using the tokenizer.")

    def encode(self, text):
        self._require_vocab()
        return [self.vocab_map.get(t, self.vocab_map["<UNK>"]) for t in tokenize(text)]

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
            json.dumps({"version": 1, "tokens": tokens}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported tokenizer file format.")
        tokens = data.get("tokens")
        if (
            not isinstance(tokens, list)
            or not all(isinstance(token, str) for token in tokens)
            or len(tokens) != len(set(tokens))
            or tokens[: len(RESERVED_TOKENS)] != RESERVED_TOKENS
        ):
            raise ValueError("Invalid tokenizer vocabulary or reserved tokens.")
        tokenizer = cls()
        tokenizer.vocab_map = {token: i for i, token in enumerate(tokens)}
        tokenizer.id_to_token = dict(enumerate(tokens))
        return tokenizer


def main():
    tokenizer = Tokenizer()
    tokenizer.build_vocab()
    tokenizer.save(PROCESSED_DIR.parent / "tokenizer.json")
    for f in sorted(TRAIN_DIR.glob("*.txt")):
        with f.open("r", encoding="utf-8") as lyric_file:
            encoded = tokenizer.encode(lyric_file.read())
            print(encoded)
            decoded = tokenizer.decode(encoded)
            print(decoded)


if __name__ == "__main__":
    main()
