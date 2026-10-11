"""Byte-level BPE, fitted locally on training songs only."""

import json
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, trainers, pre_tokenizers, decoders

from .tokenizer import RESERVED_TOKENS, TRAIN_DIR, tokenize


class BPETokenizer:
    def __init__(self, vocab_size=2000, section_tokens=True):
        self.token_level = "bpe"
        self.section_tokens = section_tokens
        self.lowercase = True
        self.special_tokens = RESERVED_TOKENS if section_tokens else RESERVED_TOKENS[:5]
        minimum = 256 + len(self.special_tokens)
        if not isinstance(section_tokens, bool):
            raise ValueError("section_tokens must be a boolean.")
        if not isinstance(vocab_size, int) or vocab_size < minimum:
            raise ValueError(
                f"BPE vocab_size must be at least {minimum} (bytes plus reserved tokens)."
            )
        self.vocab_size = vocab_size
        self.tokenizer = HFTokenizer(models.BPE(unk_token="<UNK>"))

        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def _normalize(self, text):
        # Lowercase text while keeping reserved markers atomic and uppercase.
        return "".join(tokenize(text, "char", self.section_tokens, self.lowercase))

    def train(self, songs, min_freq=2):
        if min_freq < 1:
            raise ValueError("min_freq must be positive.")
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_freq,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )

        self.tokenizer.train_from_iterator(
            (self._normalize(song) for song in songs),
            trainer=trainer,
        )

    def build_vocab(self, processed_dir=TRAIN_DIR, min_freq=None):
        paths = sorted(Path(processed_dir).glob("*.txt"))
        if not paths:
            raise ValueError(f"No training songs found in {processed_dir}.")
        self.train(
            (path.read_text(encoding="utf-8") for path in paths),
            min_freq=2 if min_freq is None else min_freq,
        )

    @property
    def vocab_map(self):
        return self.tokenizer.get_vocab()

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id("<PAD>")

    @property
    def bos_token_id(self):
        return self.tokenizer.token_to_id("<BOS>")

    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<EOS>")

    @property
    def id_to_token(self):
        return [
            self.tokenizer.id_to_token(i)
            for i in range(self.tokenizer.get_vocab_size())
        ]

    @property
    def backend_state(self):
        # Merges and preprocessing affect encoding even when token IDs match.
        return json.loads(self.tokenizer.to_str())

    def _require_vocab(self):
        if not self.vocab_map:
            raise ValueError("Train or load BPE before using the tokenizer.")

    def encode(self, text, add_boundaries=False):
        self._require_vocab()
        ids = self.tokenizer.encode(self._normalize(text)).ids
        # Prepared songs already contain boundaries; prompts need neither.
        return [self.bos_token_id, *ids, self.eos_token_id] if add_boundaries else ids

    def decode(self, ids):
        self._require_vocab()
        ids = [
            int(i)
            for i in ids
            if int(i) not in (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        ]
        return self.tokenizer.decode(ids, skip_special_tokens=False).replace(
            "<NEWLINE>", "\n"
        )

    def save(self, path):
        self._require_vocab()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "token_level": self.token_level,
                    "section_tokens": self.section_tokens,
                    "lowercase": self.lowercase,
                    "vocab_size": self.vocab_size,
                    "tokens": self.id_to_token,
                    "backend": self.backend_state,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 3 or data.get("token_level") != "bpe":
            raise ValueError("Unsupported BPE tokenizer file format.")
        instance = cls(data["vocab_size"], data["section_tokens"])
        if not isinstance(data.get("lowercase"), bool):
            raise ValueError("Invalid BPE lowercase setting.")
        instance.lowercase = data["lowercase"]
        instance.tokenizer = HFTokenizer.from_str(json.dumps(data["backend"]))
        if (
            instance.id_to_token != data["tokens"]
            or instance.id_to_token[: len(instance.special_tokens)]
            != instance.special_tokens
        ):
            raise ValueError("Invalid BPE vocabulary or reserved tokens.")
        return instance
