import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ghostwriter.cluster_contrastive import embed_song_views, visualize
from ghostwriter.data.tokenizer import create_tokenizer
from ghostwriter.models.transformer import LyricsTransformer
from ghostwriter.training.checkpoint import save_model
from ghostwriter.training.contrastive_common import (
    evaluate_pairs,
    load_contrastive,
    load_evaluation_split,
)
from ghostwriter.training.train_contrastive import train


class ContrastivePipelineTests(unittest.TestCase):
    def test_train_save_reload_evaluate_and_visualize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            for split, count in (("train", 4), ("val", 4), ("test", 6)):
                folder = processed / split
                folder.mkdir(parents=True)
                for index in range(count):
                    artist = "alpha" if index % 2 else "beta"
                    lyrics = f"<BOS> {' '.join([f'{artist} lyric {index}'] * 24)} <EOS>"
                    (folder / f"{artist}__song_{index}.txt").write_text(
                        lyrics, encoding="utf-8"
                    )
            tokenizer = create_tokenizer("word", False)
            tokenizer.build_vocab(processed / "train", min_freq=1)
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(tokenizer_path)
            model = LyricsTransformer(
                len(tokenizer.vocab_map), embedding_size=8, max_seq_len=64, dropout=0
            )
            base_path = root / "best_transformer.pt"
            save_model(
                model,
                tokenizer,
                base_path,
                "transformer",
                {"embedding_size": 8, "max_seq_len": 64, "dropout": 0},
                64,
                1,
            )
            model.context_length = 64
            output_dir = root / "contrastive" / "example"
            prior_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    report = train(
                        root,
                        output_dir,
                        model,
                        tokenizer,
                        base_path,
                        tokenizer_path,
                        "cpu",
                        epochs=1,
                        batch_size=2,
                        min_view_len=16,
                    )
                self.assertEqual(report["best_epoch"], 1)
                self.assertTrue((output_dir / "best_projection.pt").exists())
                self.assertTrue((output_dir / "history.json").exists())
                self.assertTrue((output_dir / "metrics.json").exists())
                saved_config = json.loads((output_dir / "config.json").read_text())
                self.assertEqual(saved_config["context_length"], 64)
                self.assertEqual(report["test"]["pooled"]["anchors"], 12)
                self.assertEqual(report["test"]["projected"]["anchors"], 12)
                loaded_model, loaded_tokenizer, head, config, saved = load_contrastive(
                    output_dir, "cpu"
                )
                self.assertEqual(saved["epoch"], 1)
                dataset, names = load_evaluation_split(config, "test", loaded_tokenizer)
                self.assertEqual(len(names), 6)
                first = evaluate_pairs(loaded_model, head, dataset, 2, "cpu", 0.1, 42)
                second = evaluate_pairs(loaded_model, head, dataset, 2, "cpu", 0.1, 42)
                self.assertEqual(first, second)
                self.assertEqual(first, report["test"])
                sample = (processed / "test" / "alpha__song_1.txt").read_text()
                pooled, projected = embed_song_views(
                    loaded_model, loaded_tokenizer, head, sample, "cpu", 64
                )
                self.assertEqual(pooled.shape, (8,))
                self.assertEqual(projected.shape, (8,))
                self.assertAlmostEqual(np.linalg.norm(pooled), 1, places=5)
                self.assertAlmostEqual(np.linalg.norm(projected), 1, places=5)
                vectors = np.stack([pooled, projected, pooled + 0.01, projected + 0.01])
                with patch("ghostwriter.cluster_contrastive.umap.UMAP") as reducer:
                    reducer.return_value.fit_transform.return_value = np.array(
                        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
                    )
                    chart = output_dir / "chart.html"
                    with contextlib.redirect_stdout(io.StringIO()):
                        visualize(
                            vectors,
                            ["a__1", "b__1", "a__2", "b__2"],
                            ["test"] * 4,
                            "pooled",
                            chart,
                            2,
                        )
                    self.assertTrue(chart.is_file())
            finally:
                torch.set_num_threads(prior_threads)


if __name__ == "__main__":
    unittest.main()
