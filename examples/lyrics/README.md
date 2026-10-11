# Synthetic lyric examples

These ten UTF-8 `.txt` files contain newly written synthetic text for
Ghostwriter's educational demonstration. They are not a third-party lyric
dataset and are not attributed to any real artist.

Each file represents one song, with optional section labels and blank lines
between stanzas. Copy them into `data/raw/` when you want to try the local
workflow. Ten files produce eight training songs and one song in each held-out
split with the current splitting recipe.

This is a tiny toy corpus for inspecting tokenization, tensor shapes, training,
checkpoint loading, and generation. Its scores do not establish useful lyric
quality or reliable differences between architectures. Repeated lines are
intentional toy patterns; they also make these examples unsuitable as a
generalization benchmark.
