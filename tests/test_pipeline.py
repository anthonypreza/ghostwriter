import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ghostwriter.data.tokenizer import Tokenizer
from ghostwriter.training.checkpoint import load_model
from ghostwriter import pipeline


class PipelineTests(unittest.TestCase):
    def make_songs(self, raw):
        raw.mkdir()
        for index in range(10):
            (raw / f"song_{index:02d}.txt").write_text(
                f"[Verse]\nhello world song{index}\nhello world\n", encoding="utf-8"
            )

    def test_complete_cli_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, run = root / "raw", root / "run"
            self.make_songs(raw)
            model_config = root / "gru.json"
            model_config.write_text(
                json.dumps({"embedding_size": 8, "hidden_size": 12, "dropout": 0.0})
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
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--seq-len",
                    "8",
                    "--stride",
                    "4",
                    "--min-freq",
                    "1",
                    "--device",
                    "cpu",
                    "--model",
                    "gru",
                    "--model-config",
                    str(model_config),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((run / "processed/splits.json").read_text())
            self.assertEqual(
                [len(manifest[s]) for s in ("train", "val", "test")], [8, 1, 1]
            )
            tokenizer = Tokenizer.load(run / "tokenizer.json")
            for split in ("val", "test"):
                filename = Path(manifest[split][0]).stem
                index = int(filename.split("_")[1])
                self.assertNotIn(f"song{index}", tokenizer.vocab_map)
            metrics = json.loads((run / "metrics.json").read_text())
            self.assertEqual(metrics["epochs_completed"], 1)
            self.assertEqual(metrics["best_epoch"], 1)
            for split in ("val", "test"):
                song = run / "processed" / manifest[split][0]
                expected_targets = (
                    len(tokenizer.encode(song.read_text(encoding="utf-8"))) - 1
                )
                self.assertEqual(
                    metrics["evaluation"][split]["target_tokens"], expected_targets
                )
            model, _ = load_model(run / "best_gru.pt", run / "tokenizer.json")
            self.assertEqual(model.embedding.embedding_dim, 8)
            self.assertEqual(model.gru.hidden_size, 12)
            self.assertEqual(model.context_length, 8)
            self.assertTrue((run / "config.json").exists())
            self.assertTrue((run / "history.json").exists())
            self.assertIn("[4/4]", (run / "pipeline.log").read_text())
            # A second invocation cannot overwrite a previous run.
            repeated = subprocess.run(
                result.args, capture_output=True, text=True, timeout=30
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)

    def test_stage_failure_stops_pipeline_and_is_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, run = root / "raw", root / "run"
            self.make_songs(raw)
            with (
                patch.object(
                    sys,
                    "argv",
                    ["pipeline", "--raw-dir", str(raw), "--run-dir", str(run)],
                ),
                patch.object(
                    pipeline,
                    "prepare_corpus",
                    side_effect=ValueError("synthetic failure"),
                ),
                patch.object(pipeline, "train") as training,
                patch.object(pipeline, "evaluate") as evaluation,
            ):
                with self.assertRaisesRegex(ValueError, "synthetic failure"):
                    pipeline.main()
                training.assert_not_called()
                evaluation.assert_not_called()
            self.assertIn(
                "FAILED: ValueError: synthetic failure",
                (run / "pipeline.log").read_text(),
            )
            self.assertFalse((run / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
