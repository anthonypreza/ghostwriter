import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import Tokenizer

SEQUENCE_LEN = 64
STRIDE = 32
PAD_TOKEN_ID = 0


class LyricsDataset(Dataset):
    def __init__(
        self,
        encoded_songs: list[list[int]],
        seq_len: int = SEQUENCE_LEN,
        stride: int = STRIDE,
        pad_token_id: int = PAD_TOKEN_ID,
        drop_last: bool = False,
    ) -> None:
        self.encoded_songs = encoded_songs
        self.seq_len = seq_len
        self.stride = stride
        self.pad_token_id = pad_token_id
        self.drop_last = drop_last

        self.windows = []
        for song_idx, song in enumerate(self.encoded_songs):
            for start_idx in range(0, len(song) - 1, self.stride):
                remaining = len(song) - start_idx - 1

                if self.drop_last and remaining < self.seq_len:
                    break

                self.windows.append((song_idx, start_idx))

    @classmethod
    def from_dir(
        cls,
        directory: str | Path,
        tokenizer: "Tokenizer",
        seq_len: int = SEQUENCE_LEN,
        stride: int = STRIDE,
        pad_token_id: int = PAD_TOKEN_ID,
        drop_last: bool = False,
    ) -> "LyricsDataset":
        """Encode each song separately using an already fitted tokenizer."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Song directory does not exist: {directory}")
        song_paths = sorted(path for path in directory.glob("*.txt") if path.is_file())
        if not song_paths:
            raise ValueError(f"No .txt song files found in {directory}")
        encoded_songs = [
            tokenizer.encode(path.read_text(encoding="utf-8")) for path in song_paths
        ]
        return cls(
            encoded_songs=encoded_songs,
            seq_len=seq_len,
            stride=stride,
            pad_token_id=pad_token_id,
            drop_last=drop_last,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        song_idx, start_idx = self.windows[idx]
        song = self.encoded_songs[song_idx]

        x = song[start_idx : start_idx + self.seq_len]
        y = song[start_idx + 1 : start_idx + self.seq_len + 1]

        x += [self.pad_token_id] * (self.seq_len - len(x))
        y += [self.pad_token_id] * (self.seq_len - len(y))

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def main():
    songs = [
        [2, 10, 11, 12, 13, 3],
        [2, 20, 21, 22, 23, 3],
    ]

    dataset = LyricsDataset(
        encoded_songs=songs,
        seq_len=4,
        stride=4,
        pad_token_id=0,
    )

    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"Window {i}:")
        print("X:", x.tolist())
        print("Y:", y.tolist())


if __name__ == "__main__":
    main()
