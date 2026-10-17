import argparse
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..models.registry import add_model_arguments
from ..training.checkpoint import load_model

SEQUENCE_LEN = 64
STRIDE = 32
PAD_TOKEN_ID = 0
BATCH_SIZE = 32


def song_sampler(encoded_song, min_view_len=64, max_view_len=128):
    token_count = len(encoded_song)
    if token_count // 2 < min_view_len:
        raise ValueError(
            f"[ERROR] Song is not long enough for two views of at least {min_view_len} tokens, skipping..."
        )

    view_len = min(max_view_len, token_count // 2)
    # Reserve room for one complete view on each side of the split.
    split = torch.randint(view_len, token_count - view_len + 1, (1,)).item()
    left_start = torch.randint(0, split - view_len + 1, (1,)).item()
    right_start = torch.randint(split, token_count - view_len + 1, (1,)).item()
    return (
        encoded_song[left_start : left_start + view_len],
        encoded_song[right_start : right_start + view_len],
        view_len,
    )


class ContrastiveSongDataset(Dataset):
    def __init__(
        self,
        encoded_songs: list[list[int]],
        max_seq_len: int = SEQUENCE_LEN * 2,
        pad_token_id: int = PAD_TOKEN_ID,
        min_view_len: int = SEQUENCE_LEN,
    ) -> None:
        if min_view_len < 1 or max_seq_len < min_view_len:
            raise ValueError(
                "View lengths must satisfy 0 < min_view_len <= max_seq_len."
            )
        cleaned_songs = []
        for song in encoded_songs:
            cleaned_songs.append([t for t in song if t not in (2, 3)])

        self.encoded_songs = [
            song for song in cleaned_songs if len(song) >= 2 * min_view_len
        ]

        self.max_seq_len = max_seq_len
        self.min_view_len = min_view_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.encoded_songs)

    def __getitem__(self, index) -> Any:
        encoded_song = self.encoded_songs[index]
        view1, view2, view_len = song_sampler(
            encoded_song,
            min_view_len=self.min_view_len,
            max_view_len=self.max_seq_len,
        )
        if view_len < self.max_seq_len:
            view1 = view1 + [self.pad_token_id] * (self.max_seq_len - len(view1))
            view2 = view2 + [self.pad_token_id] * (self.max_seq_len - len(view2))
        return (
            torch.tensor(view1),
            torch.tensor(view2),
            torch.tensor([t != self.pad_token_id for t in view1]),
            torch.tensor([t != self.pad_token_id for t in view2]),
        )


def load_split(processed_dir, split, tokenizer, max_seq_len=128, min_view_len=64):
    """Load one existing song split and retain names for later evaluation."""
    paths = sorted((Path(processed_dir) / split).glob("*.txt"))
    if not paths:
        raise ValueError(f"No songs found in {Path(processed_dir) / split}")
    encoded = [tokenizer.encode(path.read_text(encoding="utf-8")) for path in paths]
    names = [
        path.stem
        for path, song in zip(paths, encoded)
        if sum(token not in (2, 3) for token in song) >= 2 * min_view_len
    ]
    dataset = ContrastiveSongDataset(
        encoded,
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        min_view_len=min_view_len,
    )
    return dataset, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="New output folder; defaults to runs/<timestamp>.",
        required=True,
    )
    parser.add_argument(
        "--device", default=None, help="Default: CUDA if available, otherwise CPU."
    )
    args = parser.parse_args()
    tokenizer_path = args.run_dir / "tokenizer.json"
    checkpoint = args.run_dir / f"best_{args.model}.pt"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, tokenizer = load_model(checkpoint, tokenizer_path, device)
    processed = args.run_dir / "processed"

    encoded_songs = []
    for fname in os.listdir(Path(processed / "train")):
        with open(Path(processed / "train" / fname), "r", encoding="utf-8") as f:
            tokens = tokenizer.encode(f.read())
            encoded_songs.append(tokens)
    ds = ContrastiveSongDataset(encoded_songs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for v1, v2, m1, m2 in loader:
        views = torch.concat([v1, v2], dim=0)
        print(views.size())

        masks = torch.concat([m1, m2], dim=0)
        print(masks.size())

        B = v1.size(0)
        targets = torch.concat([torch.arange(B, 2 * B), torch.arange(0, B)])

        print(targets)


if __name__ == "__main__":
    main()
