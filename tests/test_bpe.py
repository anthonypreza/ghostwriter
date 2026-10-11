import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from ghostwriter.data.bpe_tokenizer import BPETokenizer
from ghostwriter.data.prepare import clean_lyrics
from ghostwriter.data.tokenizer import Tokenizer, create_tokenizer
from ghostwriter.inference.generate import generate
from ghostwriter.models.registry import build_model
from ghostwriter.training.checkpoint import load_model, save_model
from ghostwriter.training.summarize import inspect_run


class BPETests(unittest.TestCase):
    def test_round_trip_boundaries_unicode_and_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested/tokenizer.json"
            for sections in (True, False):
                with self.subTest(sections=sections):
                    tokenizer = create_tokenizer("bpe", sections, 400)
                    text = clean_lyrics("[Verse]\nA  can't, É!\n\nNext?", sections)
                    tokenizer.train([text], min_freq=1)
                    ids = tokenizer.encode(text)
                    self.assertEqual(ids.count(tokenizer.bos_token_id), 1)
                    self.assertEqual(ids.count(tokenizer.eos_token_id), 1)
                    expected = text.replace("<BOS>", "").replace("<EOS>", "")
                    expected = expected.replace("A  can't, É!", "a  can't, é!").replace(
                        "Next?", "next?"
                    )
                    self.assertEqual(tokenizer.decode(ids), expected)
                    # Full byte alphabet handles characters absent from training.
                    unseen = "雪 🖤"
                    self.assertEqual(tokenizer.decode(tokenizer.encode(unseen)), unseen)
                    self.assertNotIn(
                        tokenizer.vocab_map["<UNK>"], tokenizer.encode(unseen)
                    )
                    self.assertEqual("<VERSE>" in tokenizer.vocab_map, sections)
                    if sections:
                        self.assertEqual(
                            tokenizer.encode("<VERSE>"),
                            [tokenizer.vocab_map["<VERSE>"]],
                        )
                    tokenizer.save(path)
                    loaded = Tokenizer.load(path)
                    self.assertIsInstance(loaded, BPETokenizer)
                    self.assertEqual(loaded.encode(text), ids)
                    self.assertEqual(loaded.decode(ids), expected)
                    self.assertEqual(loaded.backend_state, tokenizer.backend_state)
            with self.assertRaises(ValueError):
                create_tokenizer("bpe", vocab_size=100)

    def test_checkpoint_generation_and_backend_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = BPETokenizer(300, section_tokens=False)
            tokenizer.train([clean_lyrics("A  bright sky\n\nShines!", False)])
            tokenizer.save(root / "tokenizer.json")
            model = build_model(
                "gru",
                len(tokenizer.vocab_map),
                {"embedding_size": 8, "hidden_size": 12},
            )
            save_model(
                model, tokenizer, root / "model.pt", "gru", model.model_config, 16, 1
            )
            model, loaded = load_model(root / "model.pt", root / "tokenizer.json")
            prompt = "A  sky\n\nÉ!"
            logits = torch.full((1, 16, len(loaded.vocab_map)), -100.0)
            logits[:, :, loaded.eos_token_id] = 100.0
            with patch.object(model, "forward", return_value=logits) as forward:
                text = generate(
                    model, loaded, prompt=prompt, temperature=0, max_new_tokens=10
                )
            self.assertEqual(text, prompt.lower())
            input_ids = forward.call_args.args[0][0].tolist()
            self.assertEqual(input_ids[0], loaded.bos_token_id)
            self.assertNotIn(loaded.eos_token_id, input_ids)
            # Same IDs, different encoding rules must not silently load.
            data = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
            data["backend"]["pre_tokenizer"]["add_prefix_space"] = True
            (root / "different.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "BPE merges or preprocessing"):
                load_model(root / "model.pt", root / "different.json")

    def test_complete_pipeline_and_run_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, run = root / "raw", root / "run"
            raw.mkdir()
            for i in range(10):
                (raw / f"song{i}.txt").write_text(
                    f"[Verse]\nPaper sky {i}\n\nLittle light\n", encoding="utf-8"
                )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ghostwriter.pipeline",
                    "--raw-dir",
                    str(raw),
                    "--run-dir",
                    str(run),
                    "--token-level",
                    "bpe",
                    "--vocab-size",
                    "300",
                    "--no-section-tokens",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--seq-len",
                    "8",
                    "--stride",
                    "4",
                    "--embedding-size",
                    "8",
                    "--hidden-size",
                    "12",
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model, tokenizer = load_model(run / "best_gru.pt", run / "tokenizer.json")
            # Refit solely on saved training songs: validation/test never fit merges.
            expected = BPETokenizer(300, False)
            expected.build_vocab(run / "processed/train")
            self.assertEqual(expected.backend_state, tokenizer.backend_state)
            self.assertEqual(model.context_length, 8)
            summary = inspect_run(run)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["vocab_size"], len(tokenizer.vocab_map))
            self.assertNotIn("<INTRO>", summary["commands"]["generate"])
            self.assertIsNotNone(summary["group"])
            # BPE encoding changes must create a separate comparison group.
            path = run / "tokenizer.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["backend"]["pre_tokenizer"]["add_prefix_space"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertNotEqual(inspect_run(run)["group"], summary["group"])


if __name__ == "__main__":
    unittest.main()
