import importlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ghostwriter.data.tokenizer import create_tokenizer

training = importlib.import_module("ghostwriter.training.train")


class ControlledLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.calls = 0

    def forward(self, predictions, targets):
        self.calls += 1
        # One training and one validation batch per epoch; only epoch 1 is best.
        value = 1.0 if self.calls == 2 else 2.0
        return predictions.sum() * 0 + value


class SchedulerTests(unittest.TestCase):
    def test_plateau_reduction_gets_full_patience_then_stops_at_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "song.txt").write_text("hello world", encoding="utf-8")
            tokenizer = create_tokenizer("word", False)
            tokenizer.build_vocab(root, min_freq=1)
            tokenizer.save(root / "tokenizer.json")
            dataset = [(torch.tensor([2, 2]), torch.tensor([3, 3]))]
            for scheduler, expected_epochs in (("plateau", 9), ("none", 6)):
                with (
                    self.subTest(scheduler=scheduler),
                    patch.object(
                        training.LyricsDataset, "from_dir", return_value=dataset
                    ),
                    patch.object(training.nn, "CrossEntropyLoss", ControlledLoss),
                    patch.object(training, "save_model") as save,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    result = training.train(
                        tokenizer_path=root / "tokenizer.json",
                        checkpoint_path=root / "best.pt",
                        epochs=20,
                        patience=5,
                        device_name="cpu",
                        model_config={"embedding_size": 4, "hidden_size": 4},
                        lr_scheduler=scheduler,
                        min_lr=0.00015,
                    )
                    self.assertEqual(result["epochs_completed"], expected_epochs)
                    self.assertEqual(result["best_epoch"], 1)
                    save.assert_called_once()
                    rates = result["history"]["learning_rate"]
                    if scheduler == "plateau":
                        self.assertEqual(rates, [0.0003] * 4 + [0.00015] * 5)
                        self.assertEqual(
                            result["history"]["next_learning_rate"][3], 0.00015
                        )
                        self.assertEqual(
                            output.getvalue().count("Learning rate reduced:"), 1
                        )
                    else:
                        self.assertEqual(rates, [0.0003] * 6)
                        self.assertNotIn("Learning rate reduced:", output.getvalue())

    def test_invalid_scheduler_settings(self):
        for overrides in (
            {"lr_factor": 1.0},
            {"lr_factor": float("nan")},
            {"lr_patience": -1},
            {"lr_patience": 5},
            {"min_lr": 1.0},
            {"min_lr": float("inf")},
        ):
            settings = dict(
                lr_scheduler="plateau",
                lr_factor=0.5,
                lr_patience=2,
                min_lr=1e-6,
                learning_rate=0.0003,
                patience=5,
            )
            settings.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                training.validate_scheduler_settings(**settings)


if __name__ == "__main__":
    unittest.main()
