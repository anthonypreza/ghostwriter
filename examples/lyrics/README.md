# Synthetic lyric examples

These ten UTF-8 text files contain synthetic writing for Ghostwriter's
demonstration. They are not attributed to any real artist.

Each file represents one song, with optional section labels and blank lines
between stanzas. From the repository root, this tracked directory can be passed
directly to the pipeline with `--raw-dir examples/lyrics`. The splitting recipe
produces eight training songs and one song in each held-out split.

The examples support exploring tokenization, tensor shapes, training, and
generation. Repeated lines are intentional toy patterns; results on this small
corpus do not establish useful writing quality or architecture rankings.
