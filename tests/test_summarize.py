import json
from pathlib import Path
import tempfile
import unittest

from ghostwriter.training.summarize import inspect_run, rank_runs, ps_quote


class SummaryTests(unittest.TestCase):
    def make_run(self, root, name, ppl, word="example", vocabulary=None):
        run = root / name
        (run / "processed/val").mkdir(parents=True)
        (run / "processed/val/song.txt").write_text(word, encoding="utf-8")
        documents = {
            "config.json": {
                "model": "gru",
                "seq_len": 32,
                "model_settings": {"hidden_size": 8},
            },
            "tokenizer.json": {"tokens": vocabulary or ["<PAD>", "<UNK>", "example"]},
            "metrics.json": {
                "best_epoch": 2,
                "epochs_completed": 3,
                "evaluation": {
                    "val": {"loss": 2.0, "perplexity": ppl, "accuracy": 0.2},
                    "test": {"loss": 2.1, "perplexity": ppl + 1, "accuracy": 0.1},
                },
            },
        }
        for filename, data in documents.items():
            (run / filename).write_text(json.dumps(data), encoding="utf-8")
        # Existence is sufficient for command construction; no weights are loaded.
        (run / "best_gru.pt").touch()
        return run

    def test_rank_only_comparable_runs_and_use_each_runs_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = inspect_run(self.make_run(root, "a", 10))
            b = inspect_run(self.make_run(root, "b", 8))
            other_data = inspect_run(
                self.make_run(root, "other-data", 2, word="different")
            )
            other_vocab = inspect_run(
                self.make_run(
                    root, "other-vocab", 1, vocabulary=["<PAD>", "<UNK>", "different"]
                )
            )
            rank_runs([a, b, other_data, other_vocab], "val-ppl")
            self.assertEqual(a["group"], b["group"])
            self.assertEqual((a["rank"], b["rank"]), (2, 1))
            self.assertNotEqual(other_data["group"], a["group"])
            self.assertNotEqual(other_vocab["group"], a["group"])
            self.assertIn(str(root / "a" / "best_gru.pt"), a["commands"]["generate"])
            self.assertIn(str(root / "a" / "processed"), a["commands"]["evaluate"])
            # The same token IDs can have different semantics in character mode.
            char_path = self.make_run(root, "char", 3)
            tokenizer_path = char_path / "tokenizer.json"
            tokenizer_data = json.loads(tokenizer_path.read_text())
            tokenizer_data.update(token_level="char", section_tokens=False)
            tokenizer_path.write_text(json.dumps(tokenizer_data))
            char_run = inspect_run(char_path)
            self.assertNotEqual(char_run["group"], a["group"])
            self.assertNotIn("<INTRO>", char_run["commands"]["generate"])
            self.assertIn("--max-new-tokens 1000", char_run["commands"]["generate"])

    def test_missing_corrupt_and_failed_runs_are_not_ranked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.make_run(root, "incomplete", 10)
            (run / "metrics.json").write_text('{"unfinished":', encoding="utf-8")
            (run / "pipeline.log").write_text(
                "FAILED: synthetic error", encoding="utf-8"
            )
            result = inspect_run(run)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(
                any("Cannot read metrics.json" in note for note in result["notes"])
            )
            self.assertIsNone(rank_runs([result], "val-ppl")[0]["rank"])
            (run / "best_gru.pt").unlink()
            self.assertFalse(inspect_run(run)["commands"])

    def test_powershell_quoting(self):
        self.assertEqual(ps_quote("C:/a'b/$model.pt"), "'C:/a''b/$model.pt'")


if __name__ == "__main__":
    unittest.main()
