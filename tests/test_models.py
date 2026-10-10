import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ghostwriter.data.tokenizer import Tokenizer
from ghostwriter.inference.generate import generate
from ghostwriter.models.registry import MODEL_REGISTRY, build_model
from ghostwriter.training.checkpoint import load_model, save_model, resolve_context_length


class TinyLanguageModel(nn.Module):
    """A different parameter layout verifies loading is not GRU-specific."""

    def __init__(self, vocab_size, width=4):
        super().__init__()
        self.lookup = nn.Embedding(vocab_size, width)
        self.projection = nn.Linear(width, vocab_size)

    def forward(self, inputs):
        return self.projection(self.lookup(inputs))


class ModelTests(unittest.TestCase):
    def test_registered_model_checkpoint_generation_and_tokenizer_identity(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(MODEL_REGISTRY, {"tiny": TinyLanguageModel}),
        ):
            root = Path(temporary)
            (root / "song.txt").write_text("hello world hello world", encoding="utf-8")
            tokenizer = Tokenizer()
            tokenizer.build_vocab(root, min_freq=1)
            tokenizer.save(root / "tokenizer.json")
            model = build_model("tiny", len(tokenizer.vocab_map), {"width": 7})
            sample = torch.tensor([tokenizer.encode("hello world")])
            model.eval()
            expected = model(sample)
            save_model(model, tokenizer, root / "model.pt", "tiny", {"width": 7}, 8, 1)
            restored, loaded_tokenizer = load_model(
                root / "model.pt", root / "tokenizer.json"
            )
            self.assertIsInstance(restored, TinyLanguageModel)
            self.assertTrue(torch.equal(expected, restored(sample)))
            self.assertEqual(restored.context_length, 8)
            self.assertEqual(resolve_context_length(restored), 8)
            with self.assertRaisesRegex(ValueError, "trained with context length 8"):
                generate(restored, loaded_tokenizer, context_length=16, max_new_tokens=0)
            self.assertIsInstance(
                generate(restored, loaded_tokenizer, max_new_tokens=2), str
            )
            # A long generation slides over exactly the trained context size.
            with torch.no_grad():
                restored.projection.weight.zero_()
                restored.projection.bias.zero_()
                restored.projection.bias[loaded_tokenizer.vocab_map["hello"]] = 10
            observed_lengths = []
            hook = restored.register_forward_pre_hook(
                lambda _model, inputs: observed_lengths.append(inputs[0].size(1))
            )
            generate(restored, loaded_tokenizer, max_new_tokens=12, temperature=0)
            hook.remove()
            self.assertEqual(max(observed_lengths), 8)
            self.assertEqual(observed_lengths[-1], 8)
            saved_tokens = json.loads((root / "tokenizer.json").read_text())
            saved_tokens["tokens"][-2:] = saved_tokens["tokens"][-2:][::-1]
            (root / "wrong-tokenizer.json").write_text(json.dumps(saved_tokens))
            with self.assertRaisesRegex(ValueError, "token IDs"):
                load_model(root / "model.pt", root / "wrong-tokenizer.json")

    def test_legacy_gru_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "song.txt").write_text("hello hello", encoding="utf-8")
            tokenizer = Tokenizer()
            tokenizer.build_vocab(root)
            tokenizer.save(root / "tokenizer.json")
            original = build_model(
                "gru",
                len(tokenizer.vocab_map),
                {"embedding_size": 8, "hidden_size": 12},
            )
            torch.save(original.state_dict(), root / "legacy.pt")
            restored, _ = load_model(root / "legacy.pt", root / "tokenizer.json")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                resolve_context_length(restored)
            self.assertEqual(resolve_context_length(restored, 64), 64)
            (root / "config.json").write_text(json.dumps({"seq_len": 64}))
            configured, _ = load_model(root / "legacy.pt", root / "tokenizer.json")
            self.assertEqual(resolve_context_length(configured), 64)
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_invalid_model_selection(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model("missing", 20)
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            build_model("gru", 20, {"vocab_size": 30})


if __name__ == "__main__":
    unittest.main()
