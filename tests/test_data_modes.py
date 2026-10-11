import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from ghostwriter.data.prepare import clean_lyrics
from ghostwriter.data.tokenizer import RESERVED_TOKENS, Tokenizer, tokenize
from ghostwriter.inference.generate import generate, render_tokens
from ghostwriter.models.registry import build_model
from ghostwriter.training.checkpoint import load_model, save_model


class DataModeTests(unittest.TestCase):
    def test_section_removal_preserves_stanzas_and_normalizes_blank_lines(self):
        raw = "[Intro]\nHello!\n\n\n[Verse 1]\nWorld.\n[Chorus]\nAgain\n\n<OUTRO>\nBye"
        self.assertEqual(
            clean_lyrics(raw, section_tokens=False),
            "<BOS>\nHello!\n\nWorld.\n\nAgain\n\nBye\n<EOS>",
        )
        self.assertIn("<VERSE>", clean_lyrics(raw))
        self.assertEqual(
            tokenize("Can't, go!\n\n"),
            ["can't", ",", "go", "!", "<NEWLINE>", "<NEWLINE>"],
        )

    def test_all_four_tokenizer_configurations_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for level in ("word", "char"):
                for sections in (True, False):
                    with self.subTest(level=level, sections=sections):
                        text = clean_lyrics("[Verse]\nA  can't, Z!\n\nNext?", sections)
                        (root / "song.txt").write_text(text, encoding="utf-8")
                        tokenizer = Tokenizer(level, sections)
                        tokenizer.build_vocab(root)
                        tokenizer.save(root / "tokenizer.json")
                        loaded = Tokenizer.load(root / "tokenizer.json")
                        self.assertEqual(
                            (loaded.token_level, loaded.section_tokens),
                            (level, sections),
                        )
                        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
                        self.assertEqual("<VERSE>" in loaded.vocab_map, sections)
                        if level == "char":
                            expected = text.replace("<BOS>", "").replace("<EOS>", "")
                            expected = expected.replace(
                                "A  can't, Z!", "a  can't, z!"
                            ).replace("Next?", "next?")
                            self.assertEqual(
                                render_tokens(
                                    loaded.decode(loaded.encode(text)), level
                                ),
                                expected,
                            )
                            self.assertNotIn(
                                loaded.vocab_map["<UNK>"], loaded.encode(text)
                            )

    def test_legacy_tokenizer_and_checkpoint_mode_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tokenizer.json"
            path.write_text(
                json.dumps({"version": 1, "tokens": RESERVED_TOKENS + ["hello"]})
            )
            legacy = Tokenizer.load(path)
            self.assertEqual(legacy.token_level, "word")
            self.assertTrue(legacy.section_tokens)
            self.assertEqual(legacy.decode(legacy.encode("Hello")), ["hello"])

            (root / "song.txt").write_text("A  Z\n\nZ", encoding="utf-8")
            tokenizer = Tokenizer("char", False)
            tokenizer.build_vocab(root)
            tokenizer.save(path)
            model = build_model(
                "gru",
                len(tokenizer.vocab_map),
                {"embedding_size": 8, "hidden_size": 12},
            )
            save_model(
                model, tokenizer, root / "model.pt", "gru", model.model_config, 8, 1
            )
            restored, loaded = load_model(root / "model.pt", path)
            with torch.no_grad():
                for parameter in restored.parameters():
                    parameter.zero_()
                restored.output_layer.bias[loaded.vocab_map["z"]] = 10
            self.assertEqual(
                generate(
                    restored, loaded, prompt="A  \n\n", max_new_tokens=2, temperature=0
                ),
                "a\n\nzz",
            )
            self.assertEqual(generate(restored, loaded, max_new_tokens=0), "")
            self.assertEqual(len(generate(restored, loaded, temperature=0)), 1000)
            self.assertEqual(
                len(generate(restored, loaded, temperature=0, max_new_tokens=7)), 7
            )
            old_char_path = root / "old-char.json"
            old_char_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "token_level": "char",
                        "section_tokens": False,
                        "tokens": RESERVED_TOKENS[:5] + ["A", "a"],
                    }
                )
            )
            old_char = Tokenizer.load(old_char_path)
            self.assertFalse(old_char.lowercase)
            self.assertEqual(old_char.decode(old_char.encode("Aa")), ["A", "a"])
            data = json.loads(path.read_text())
            data["token_level"] = "word"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "tokenizer settings"):
                load_model(root / "model.pt", path)

    def test_character_section_free_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, run = root / "raw", root / "run"
            raw.mkdir()
            for index in range(10):
                (raw / f"song{index}.txt").write_text(
                    "[Verse]\nA  Z!\n\n[Chorus]\nAgain?", encoding="utf-8"
                )
            config = root / "model.json"
            config.write_text(json.dumps({"embedding_size": 8, "hidden_size": 12}))
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
                    "char",
                    "--no-section-tokens",
                    "--epochs",
                    "1",
                    "--seq-len",
                    "8",
                    "--stride",
                    "4",
                    "--device",
                    "cpu",
                    "--model-config",
                    str(config),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads((run / "config.json").read_text())
            self.assertEqual(saved["token_level"], "char")
            self.assertFalse(saved["section_tokens"])
            self.assertEqual(saved["min_freq"], 1)
            self.assertNotIn('--prompt "<INTRO>"', result.stdout)
            self.assertIn("--max-new-tokens 1000", result.stdout)
            model, tokenizer = load_model(run / "best_gru.pt", run / "tokenizer.json")
            self.assertEqual(tokenizer.token_level, "char")
            self.assertNotIn("<INTRO>", tokenizer.vocab_map)
            self.assertIsInstance(generate(model, tokenizer, max_new_tokens=3), str)


if __name__ == "__main__":
    unittest.main()
