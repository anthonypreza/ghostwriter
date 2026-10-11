from pathlib import Path
import re
import tempfile
import unittest

import torch
from torch import nn

from ghostwriter.data.prepare import clean_lyrics, normalize_lyrics
from ghostwriter.data.tokenizer import create_tokenizer
from ghostwriter.inference.generate import generate


class FixedLogits(nn.Module):
    def __init__(self, vocab_size, scores):
        super().__init__()
        self.scores = nn.Parameter(torch.full((vocab_size,), -torch.inf))
        with torch.no_grad():
            for token_id, value in scores.items():
                self.scores[token_id] = value
        self.context_length = 16
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return self.scores.expand(inputs.shape[0], inputs.shape[1], -1)


class NewlineTests(unittest.TestCase):
    def test_normalization_and_section_removal(self):
        self.assertEqual(
            normalize_lyrics(" \r\nHello \t\r\n \t\r\n\r\n\rWorld\t\n "),
            "Hello\n\nWorld",
        )
        self.assertEqual(normalize_lyrics("one\nline\n\nstanza"), "one\nline\n\nstanza")
        self.assertEqual(normalize_lyrics("line\n\n", strip=False), "line\n\n")
        for sections in (True, False):
            text = clean_lyrics("[Intro]\nHello\n\n\n[Verse]\n\n\nWorld", sections)
            self.assertNotIn("\n\n\n", text)
            self.assertIn("\n\n", text)

    def test_rejects_third_newline_with_greedy_and_top_k_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "song.txt").write_text("start x\n\nx", encoding="utf-8")
            for level in ("word", "char", "bpe"):
                tokenizer = create_tokenizer(
                    level, False, 300 if level == "bpe" else None
                )
                tokenizer.build_vocab(root, min_freq=1)
                for temperature in (0, 1):
                    with self.subTest(level=level, temperature=temperature):
                        model = FixedLogits(
                            len(tokenizer.vocab_map),
                            {
                                tokenizer.vocab_map["<NEWLINE>"]: 10,
                                tokenizer.vocab_map["x"]: 9,
                            },
                        )
                        text = generate(
                            model,
                            tokenizer,
                            prompt="start",
                            max_new_tokens=6,
                            temperature=temperature,
                            top_k=1,
                            seed=42,
                        )
                        self.assertEqual(text, "start\n\nx\n\nx")
                        self.assertEqual(model.calls, 6)
                        self.assertTrue(
                            all(len(run) <= 2 for run in re.findall(r"\n+", text))
                        )

    def test_bpe_candidate_containing_three_newlines_is_rejected(self):
        tokenizer = create_tokenizer("bpe", False, 300)
        tokenizer.train(["start x"])
        tokenizer.tokenizer.add_tokens(["\n\n\n"])
        invalid = tokenizer.vocab_map["\n\n\n"]
        self.assertEqual(tokenizer.decode([invalid]), "\n\n\n")
        model = FixedLogits(
            len(tokenizer.vocab_map), {invalid: 10, tokenizer.vocab_map["x"]: 9}
        )
        self.assertEqual(
            generate(model, tokenizer, prompt="start", max_new_tokens=1, top_k=1),
            "startx",
        )

    def test_no_valid_candidate_stops_without_retrying_forever(self):
        tokenizer = create_tokenizer("bpe", False, 300)
        tokenizer.train(["start"])
        model = FixedLogits(
            len(tokenizer.vocab_map), {tokenizer.vocab_map["<NEWLINE>"]: 10}
        )
        self.assertEqual(
            generate(
                model, tokenizer, prompt="start\n\n\n\n", max_new_tokens=20, top_k=1
            ),
            "start\n\n",
        )
        self.assertEqual(model.calls, 1)

    def test_existing_triple_newline_does_not_block_continuation(self):
        tokenizer = create_tokenizer("bpe", False, 300)
        tokenizer.train(["start x"])
        model = FixedLogits(len(tokenizer.vocab_map), {tokenizer.vocab_map["x"]: 10})
        # Reserved newline markers decode after prompt normalization.
        self.assertEqual(
            generate(
                model,
                tokenizer,
                prompt="start<NEWLINE><NEWLINE><NEWLINE>",
                max_new_tokens=2,
                top_k=1,
            ),
            "start\n\n\nxx",
        )
        self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
