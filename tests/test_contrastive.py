import contextlib
import io
import unittest

import torch

from ghostwriter.data.contrastive import ContrastiveSongDataset, song_sampler


class SongSamplerTests(unittest.TestCase):
    def test_views_are_disjoint_and_have_expected_length(self):
        torch.manual_seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            for token_count in (127, 128, 129, 255, 256, 500):
                song = list(range(10, token_count + 10))
                with self.subTest(token_count=token_count):
                    for _ in range(30):
                        if token_count < 128:
                            with self.assertRaises(ValueError):
                                song_sampler(song)
                            continue
                        sample = song_sampler(song)
                        left, right, view_len = sample
                        self.assertEqual(view_len, min(128, token_count // 2))
                        self.assertEqual(len(left), view_len)
                        self.assertEqual(len(right), view_len)
                        self.assertLess(left[-1], right[0])
                        self.assertTrue(set(left).isdisjoint(right))
                        self.assertTrue(set(left + right).issubset(song))

    def test_long_song_views_change_with_sampling(self):
        torch.manual_seed(42)
        song = list(range(10, 510))
        with contextlib.redirect_stdout(io.StringIO()):
            pairs = {
                (tuple(left), tuple(right))
                for left, right, _ in (song_sampler(song) for _ in range(20))
            }
        self.assertGreater(len(pairs), 1)

    def test_eligibility_uses_minimum_view_length_not_maximum(self):
        short_song = [2, *range(10, 136), 3]  # 126 usable tokens
        eligible_song = [2, *range(10, 138), 3]  # 128 usable tokens
        dataset = ContrastiveSongDataset([short_song, eligible_song], max_seq_len=256)
        self.assertEqual(len(dataset), 1)
        with contextlib.redirect_stdout(io.StringIO()):
            view1, view2, mask1, mask2 = dataset[0]
        for view, mask in ((view1, mask1), (view2, mask2)):
            self.assertEqual(tuple(view.shape), (256,))
            self.assertEqual(tuple(mask.shape), (256,))
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(mask.sum().item(), 64)
            self.assertTrue(torch.all(view[~mask] == dataset.pad_token_id))


if __name__ == "__main__":
    unittest.main()
