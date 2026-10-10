# Ghostwriter

An educational PyTorch project for training and comparing neural language models
that generate original song lyrics. Start with a small GRU baseline; add other
architectures incrementally.

## Current step: local song catalog

Maintain `data/song_catalog.csv` locally. This file and all other files under
`data/` are ignored by Git. Spotify is not used.

The CSV is UTF-8, with these columns:

| Column | Meaning |
| --- | --- |
| `artist` | Credited artist or band name |
| `album` | Release title; use the single title for a standalone single |
| `song_title` | Song title as listed on that release |
| `source_url` | Web page used to verify the release track list |

Prefer official artist/label track lists and other reliable release listings.
Keep a source link on each row so titles and album assignments can be checked.
Use normal CSV quoting for commas, quotes, and other punctuation in names.

The initial catalog uses MusicBrainz release listings, with a local coverage
summary in `data/catalog_notes.md`. It selects one official edition of each
studio album or EP and adds non-album tracks from singles. Regional bonus
editions, live releases, compilations, and remix releases are outside this
initial scope. The catalog can be edited directly as new artists are added.

The ignored local collectors support targeted additions:

```powershell
# Add a band's eligible studio albums, EPs, and standalone singles.
python data/_collect_catalog.py --artist "Slipknot"

# Add just one release, preserving other catalog entries.
python data/_collect_catalog.py --artist "Slipknot" --album "Iowa"

# Preview uncollected songs before starting collection.
python data/_collect_lyrics.py --artist "Slipknot" --album "Iowa" --dry-run
python data/_collect_lyrics.py --artist "Slipknot" --album "Iowa" --limit 100
```

`--band` is an alias for `--artist`. Names and album titles match exactly,
ignoring case and surrounding whitespace. Repeat `--artist` for several bands;
catalog album selection requires one artist, and `--album` can be repeated.
The catalog merges by artist, album, and song title, retaining existing rows
and source URLs. Ambiguous MusicBrainz artist names require disambiguation in
the collector's `KNOWN` mapping.

Lyrics selection filters the catalog first, then skips existing raw files and
deduplicates artist/song filenames across releases before random sampling.
The default limit is 100 uncollected songs; `--limit 0` selects all matches.
`--seed` defaults to 42. Dry-run makes no Genius requests or browser launches.
Neither collector automatically changes processed splits, the tokenizer, or
checkpoints. Prepare the expanded corpus separately for a new training run;
keep the original tokenizer with existing checkpoints.

Deduplicate identical `(artist, album, song_title)` entries, ignoring case and
surrounding whitespace. The same song on different releases remains separate
at this stage. Later, consolidate duplicate lyrics before splitting into
training and validation sets, so repeated songs cannot leak across splits.

The catalog contains metadata only. Lyric retrieval and tokenization are later
steps. Store lyrics locally and record their independent source and permission
for use before adding them to the training dataset. Neither lyrics nor private
trained weights belong in this repository.

## Learning sequence

### Preparing song splits

Run `python -m ghostwriter.data.prepare` to clean the local `data/raw/*.txt`
files and split whole songs into `data/processed/train/`, `val/`, and `test/`.
The split uses an 80/10/10 ratio with seed 42; counts are rounded for small
corpora. At least six song files are required.

`data/processed/splits.json` records the seed and each split's file paths,
relative to `data/processed`. Input filenames are sorted before splitting,
so unchanged inputs produce the same assignments. Adding or removing songs
can change assignments. Rerunning preparation removes stale `.txt` files
inside the three generated split folders.

Each raw file is treated as one song. Consolidate duplicate songs stored under
different filenames before preparation. The tokenizer builds its vocabulary
from the training folder by default to keep validation and test data held out.

### Evaluating and generating

### Browsing past runs

```powershell
python -m ghostwriter.training.summarize
python -m ghostwriter.training.summarize --details
python -m ghostwriter.training.summarize --model lstm --top 3
python -m ghostwriter.training.summarize --run 20261010-153144-033141
```

The summary shows model, status, best/completed epochs, split sizes, context
length, vocabulary size, validation loss/perplexity/accuracy, and test perplexity.
`--details` (or selecting one `--run`) prints model and training settings,
full validation/test metrics, elapsed time, and PowerShell commands to generate
or evaluate using the run's paired checkpoint, tokenizer, and processed files.

Each invocation also saves the displayed runs and their full details to
`runs/summary.md`, already ignored by Git. Use `--runs-dir` to scan another
folder, `--output` for a different Markdown destination, and `--color` to
control terminal colors. Rerun the command to refresh after training finishes.

Ranking defaults to lower validation perplexity; `--sort val-loss` and
`--sort val-accuracy` are also supported. Ranks are assigned separately within
groups sharing identical saved validation filenames/content, vocabulary, and
evaluation window length. They do not establish a winner across different
groups. The grouping assumes the pipeline's non-overlapping evaluation recipe.
`--top N` keeps the top N per group, plus incomplete/unranked runs. Missing or
malformed metrics are reported rather than silently omitted; unfinished runs
are not ranked. Checkpoint availability is reported separately from completion.

### Running the pipeline and using checkpoints

Run all four stages with one command:

```powershell
python -m ghostwriter.pipeline
# Shorter run or explicit hardware:
python -m ghostwriter.pipeline --epochs 10 --batch-size 32 --device cpu
# Select a model and its constructor settings:
python -m ghostwriter.pipeline --model gru --model-config configs/gru.json
```

The pipeline prints numbered stages, split sizes, vocabulary size, model and
device details, epoch metrics/timings, checkpoint improvements, and final
validation/test metrics. During longer epochs it prints batch progress about
every 30 seconds. Output is also saved to `pipeline.log`.
PowerShell output is colored automatically: cyan stages, magenta metrics,
green checkpoint saves/completion, yellow stopping notices, and red failures.
The saved log stays plain text. Use `--color never` to disable colors or
`--color always` to force them. Automatic colors respect `NO_COLOR` and are
disabled when output is redirected.

Every invocation creates a new ignored `runs/<timestamp>/` folder containing
`processed/{train,val,test}/`, `processed/splits.json`, `tokenizer.json`,
`best_<model>.pt`, `config.json`, `history.json`, `metrics.json`, and `pipeline.log`.
Use `--run-dir runs/my-experiment` to name a new run, or `--raw-dir` to select
another song folder. Existing run directories are rejected. This preserves
the tokenizer paired with each checkpoint and leaves older shared artifacts
available. The pipeline prints a generation command with the correct paths.

Other options include `--patience`, `--min-freq`, `--seq-len`, `--stride`, and
`--seed`. The seed option controls model initialization and batch shuffling;
song splitting retains seed 42. The default device is CUDA when available,
otherwise CPU. Training uses validation loss for early stopping, then final
evaluation reloads the best checkpoint and evaluates validation and test once.
The pipeline stops at the first error and records a failure summary in its log.

Training is now callable without loading data on import. Standalone training
still runs with `python -m ghostwriter.training.train`; use `--plot` to display
its learning curves. Held-out evaluation is a separate stage/command.

### Adding architectures

Both training and the pipeline accept `--model` and `--model-config`. GRU is
currently the only implemented architecture and remains the default:

```powershell
python -m ghostwriter.training.train --model gru --model-config configs/gru.json
```

The JSON file contains keyword arguments for the model constructor, such as
embedding size, hidden size, and dropout. `vocab_size` always comes from the
tokenizer. Model settings and the selected architecture are recorded with
the run and checkpoint.

To add the next architecture incrementally:

1. Implement its `nn.Module` in its own file, for example `models/lstm.py`.
2. Its constructor accepts `vocab_size` and its own configuration keywords.
3. `forward(token_ids)` takes integer IDs shaped `(batch, sequence)` and returns
   raw logits shaped `(batch, sequence, vocabulary)`. Do not apply softmax in
   the model; cross-entropy handles that during training.
4. Import the class in `models/registry.py` and add its name to `MODEL_REGISTRY`,
   for example `"lstm": LyricsLSTM`.

After registration, `--model lstm` works in both commands; evaluation and
generation recover the architecture directly from its checkpoint. RNN and
Transformer follow the same interface. The decoder-only Transformer must use
positional information and a causal attention mask, so positions cannot see
future targets. Its context capacity must support the selected `--seq-len`.
Evaluation and generation use the sequence length stored at training and reject
overrides that differ. You do not need to remember or pass the context length.

After training saves `best_gru.pt`, evaluate a split or generate text:

```powershell
python -m ghostwriter.training.eval --split test
python -m ghostwriter.inference.generate --prompt "<VERSE>" --max-new-tokens 100 --temperature 0.8 --top-k 20
```

Both commands accept `--checkpoint`, `--tokenizer`, and `--device` (default
`cpu`). Use the exact tokenizer used to train the checkpoint. New checkpoints
store architecture settings, token identities, and training sequence length;
loading rejects mismatched token IDs. Legacy weights-only GRU files still
load, but can only check vocabulary size. For context length, they use an
adjacent run `config.json` if available. Without training-length metadata,
you must supply the original length explicitly; it cannot be inferred from
the weights.

Evaluation reports mean cross-entropy, perplexity, next-token accuracy, and
non-padding target count. It uses non-overlapping windows, counting each target
once; the training loop uses overlapping windows, so its reported metrics can
differ. Evaluation resets context at each window.

Generation predicts one token, appends it, and repeats until `<EOS>` or the
token limit. It recomputes predictions over the latest tokens, capped at the
training context saved in the checkpoint. `--context-length` is
normally omitted; a different length is rejected. Short prefixes are used
as-is until enough tokens have been generated to fill the context window.
`--temperature 0` uses greedy selection;
positive values sample, with larger values producing more varied choices.
`--top-k 0` disables top-k filtering. Generation uses fresh randomness by
default; pass `--seed 42` for reproducible sampling on the same device.
Greedy generation (`--temperature 0`) stays deterministic regardless of seed.
Newline tokens become line breaks; section markers remain
visible. Padding, unknown, and beginning-of-song tokens are excluded from
generated choices.

Generation colors section markers in PowerShell automatically: blue verses,
magenta choruses, cyan pre/post-choruses, yellow bridges, and green intros/outros.
Use `--color always` or `--color never` to override automatic detection.
Redirected output and the Python `generate()` return value stay plain text.
Interactive generation also shows a spaced header/footer, model/run details,
sampling settings, and the actual random seed so you can reproduce a sample.
Lyrics are indented inside the frame. Use `--plain` to print only lyrics with
no frame or colors. Redirected output omits the frame automatically.

### Next steps

1. Curate and verify the local song catalog.
2. Build a separately sourced local lyrics dataset and manifest.
3. Implement word tokenization and next-token examples.
4. Learn embeddings, tensor shapes, and the training loop with a GRU.
5. Compare RNN, LSTM, GRU, and a decoder-only Transformer.
6. Evaluate perplexity, generation, repetition, and memorization.

The inference service and frontend come after the dataset and baseline model.
