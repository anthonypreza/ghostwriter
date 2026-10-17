# Ghostwriter

Ghostwriter is a **learning and demonstration project** for training sequence
models with PyTorch. It uses lyric-shaped text to explore next-token prediction
and compare RNN, LSTM, GRU, and causal Transformer architectures. It is an
educational experiment, not a production songwriting system or a guarantee of
original output.

**Bring your own data:** provide UTF-8 `.txt` files containing lyrics you are
permitted to use for the intended training. Keep your dataset and trained
weights local. The included files in `examples/lyrics/` contain only newly
written synthetic demonstration text.

## Problem setup

A language model predicts the next token from preceding tokens. A token can be
a word, punctuation mark, character, or control marker. Inputs and targets are
shifted by one position:

```text
Text:     lamp
Input:    l a m
Target:   a m p
```

At each position, the model produces a score for every vocabulary token.
Cross-entropy loss measures how well those scores predict the target; training
updates the model parameters to reduce that loss.

```text
Local .txt files -> preparation and song splits -> tokenization and windows
                -> model training -> held-out evaluation -> generation
```

## Local data interface

Place one song per file directly inside `data/raw/`. Use UTF-8 encoding,
ordinary line breaks for lyric lines, and blank lines between stanzas.
Filenames identify songs in the split manifest. No separate catalog is needed.
Optional section labels occupy their own lines:

```text
[Intro]
A paper satellite circles the room

[Verse]
We fold a little daylight into squares
And stack tomorrow gently on the stairs
```

```text
examples/lyrics/          Included synthetic demonstration files
data/raw/                Your local input .txt files
data/processed/
  train/                 Cleaned training songs
  val/                   Cleaned validation songs
  test/                  Cleaned test songs
  splits.json            Song assignments and preparation settings
data/tokenizer.json      Shared vocabulary and tokenization settings
runs/                    Saved experiment artifacts
```

`data/`, `runs/`, `.env`, and weights ending in `.pt`, `.pth`, or `.safetensors`
are excluded by `.gitignore`.
Only the synthetic examples are intended to be included with the code. Review
what you share, including generated text: models can memorize and reproduce
parts of their training data.

At least six song files are required. Remove duplicate songs before splitting,
including copies under different filenames, to avoid overlap between training
and held-out data.

## Setup and synthetic quick start

Use **Python 3.12** and [uv](https://docs.astral.sh/uv/). The default environment
uses CPU PyTorch on Windows and Linux, and the standard PyTorch wheel on macOS.
No GPU or private credentials are required. Supported operating systems and
processor architectures depend on available PyTorch and scientific Python
wheels; see [PyTorch installation requirements](https://pytorch.org/get-started/locally/).
The current lock targets Windows x86-64, Linux x86-64 with glibc, and Apple
Silicon Macs running macOS 14 or newer. Intel Macs and other Unix systems are
not covered by this setup.

With `uv` installed, run from the repository root in PowerShell:

```powershell
uv sync --locked
New-Item -ItemType Directory -Force data/raw | Out-Null
Copy-Item examples/lyrics/*.txt data/raw/
uv run --locked python -m ghostwriter.pipeline --model gru --epochs 3 --batch-size 8 --min-freq 1 --device cpu
```

On macOS or Linux, use a POSIX shell instead:

```sh
uv sync --locked
mkdir -p data/raw
cp examples/lyrics/*.txt data/raw/
uv run --locked python -m ghostwriter.pipeline --model gru --epochs 3 --batch-size 8 --min-freq 1 --device cpu
```

Python module commands below work in either shell. Where an example uses a
PowerShell `$run` variable, use `run="runs/REPLACE_WITH_RUN_NAME"` in a POSIX
shell. CPU execution is the portable baseline; macOS GPU acceleration is not
part of this setup.

Copy the examples only into a dataset where you want them included. The ten
synthetic files exercise the workflow; they are too small to establish useful
generation quality or reliable architecture rankings.

The pipeline performs preparation, tokenization, training, and evaluation in
one invocation. It creates a new ignored `runs/<timestamp>/` directory with
its own `processed/` splits, tokenizer, checkpoint, config, history, metrics,
and log. These copies preserve each experiment's data; the pipeline does not
overwrite the shared `data/processed/` directory. Use
`--run-dir runs/my-experiment` to name a new run. Existing run directories
are rejected.

## 1. Data preparation

To inspect preparation separately:

```powershell
uv run --locked python -m ghostwriter.data.prepare
```

This reads `data/raw/*.txt`, cleans annotations, adds beginning/end-of-song
markers, and writes `data/processed/{train,val,test}/` and `splits.json`.
Splitting happens **by song before token windows are created**, approximately
80/10/10 with seed 42. Counts are rounded for small datasets. Adding or removing
files can change assignments. Repeating preparation replaces generated files
and removes stale `.txt` files from the three split folders.

Recognized section labels become tokens such as `<VERSE>` and `<CHORUS>` by
default. `--no-section-tokens` removes labels, preserves stanza breaks,
and leaves stanza breaks where labels separated lyric lines. `<BOS>` and
`<EOS>` still mark song boundaries.

Preparation normalizes line endings, removes spaces/tabs before newlines,
and caps consecutive newlines at two (one blank line between stanzas).
This applies to newly prepared datasets; existing saved runs stay intact.

## 2. Tokenization and windows

Build the shared tokenizer from the training split only:

```powershell
uv run --locked python -m ghostwriter.data.tokenizer
```

It saves `data/tokenizer.json`. Validation and test songs use that fixed
vocabulary. Word/character tokens missing from that vocabulary map to `<UNK>`;
BPE uses a full byte alphabet to represent unseen text.

The pipeline offers two independent choices:

| Option | Behavior |
| --- | --- |
| `--token-level word` | Default: lowercase words, preserve contractions, separate punctuation |
| `--token-level char` | Lowercase characters, preserving spaces and punctuation |
| `--token-level bpe` | Lowercase byte-level BPE pieces, preserving spaces and punctuation |
| `--vocab-size 2000` | BPE target vocabulary size, including reserved tokens; default 2000 |
| `--section-tokens` | Default: include recognized section identities |
| `--no-section-tokens` | Learn continuous text and stanza breaks without section identities |

Newlines become `<NEWLINE>`, including consecutive newlines for blank lines.
Control tokens and enabled section tokens stay atomic and uppercase. Minimum
frequency defaults to 2 for words/BPE and 1 for characters. For BPE it controls
the minimum frequency for learning merges. Settings are saved with
the vocabulary; older character tokenizers retain their original casing rules.

```powershell
uv run --locked python -m ghostwriter.pipeline --model gru --token-level char --no-section-tokens --seq-len 128 --stride 64 --device cpu
```

BPE is trained locally with [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/).
Only training songs fit its vocabulary and merges; no pretrained tokenizer is
downloaded. The actual vocabulary can be smaller than the requested target.
The minimum target is 271 with sections, or 261 without sections, to include
all 256 bytes and reserved markers. For example:

```sh
uv run --locked python -m ghostwriter.pipeline --model gru --token-level bpe --vocab-size 2000 --no-section-tokens --seq-len 128 --stride 64 --device cpu
```

The run's `tokenizer.json` contains the BPE vocabulary, merges, and encoding
settings. Training, evaluation, and generation load the correct tokenizer
automatically. Generation decodes BPE pieces together to restore whitespace
and blank lines. Prepared songs already include `<BOS>`/`<EOS>`; encoding does
not add another pair.

The standalone preparation and tokenizer commands accept the section option;
tokenizer creation also accepts `--token-level`, `--min-freq`, and BPE's
`--vocab-size`. Use matching
settings for those two stages.

`LyricsDataset` creates shifted input/target windows within each song.
`--seq-len` sets tokens per input; `--stride` sets how far the next training
window advances. Short final windows are padded, and padding targets are
ignored by the loss. A context of 128 means 128 token positions, not 128 words
in character or BPE mode. Start a new run when changing the representation.

## 3. Model architecture

All models accept integer token IDs shaped `(batch, sequence)`:

| Model | Sequence processing |
| --- | --- |
| `rnn` | Basic recurrent hidden-state updates |
| `lstm` | Recurrent updates with a cell state and gates |
| `gru` | Gated recurrent updates; the default baseline |
| `transformer` | Causal self-attention with learned positional embeddings |

An embedding maps IDs to learned vectors shaped `(batch, sequence,
embedding_size)`. Sequence layers process those vectors, then a linear layer
produces raw logits shaped `(batch, sequence, vocabulary_size)`. Cross-entropy
consumes the logits directly during training.

The Transformer uses a causally masked `TransformerEncoder` stack as an
autoregressive model, with no separate source encoder or cross-attention.
Positions cannot attend to future positions. Its four attention heads require
an embedding size divisible by four. Positional capacity (`max_seq_len`) must
cover the actual training context (`--seq-len`).

Each architecture lives in its own file under `src/ghostwriter/models/`:

```powershell
uv run --locked python -m ghostwriter.pipeline --model transformer --embedding-size 256 --seq-len 128 --stride 64 --device cpu
```

`--model-config` accepts a JSON file of constructor settings, such as embedding
size, hidden size, dropout, or positional capacity. Supported keys depend on
the model. `--embedding-size`, `--hidden-size` (recurrent models), and
`--dropout` override their values from the file; vocabulary
size always comes from the tokenizer.

## 4. Training and comparison

The loop performs a forward pass, computes loss, clears gradients,
backpropagates, clips gradients, and updates parameters with AdamW. Validation
runs without gradient updates. Early stopping monitors validation loss and
keeps the best checkpoint. Recurrent hidden state resets between windows,
limiting learned within-example context to the training window.

Pipeline options include `--epochs`, `--batch-size`, `--patience`, `--seed`,
`--seq-len`, `--stride`, and `--min-freq`. The training seed controls
initialization and shuffling; the split seed stays 42. Resolved model and
training settings are recorded in each run.

For example, configure a GRU with 128-dimensional embeddings, 256 hidden
units, 128-token windows, stride 64, dropout 0.2, and section labels removed:

```sh
uv run --locked python -m ghostwriter.pipeline --model gru --embedding-size 128 --hidden-size 256 --seq-len 128 --stride 64 --dropout 0.2 --no-section-tokens --epochs 40 --patience 5
```

These are explicit run settings, rather than changes to the defaults. Epochs
are capped at 40; patience 5 stops after five consecutive epochs without a
validation-loss improvement. Standalone training also accepts the model-size,
dropout, sequence-length, and stride flags; it inherits section handling from
the already prepared files and saved tokenizer.

Both the pipeline and standalone training accept `--learning-rate` (default
`0.0003`) and `--weight-decay` (default `0.01`) for AdamW. For example, append
`--learning-rate 0.0003 --weight-decay 0.05` to a training command. Learning
rate controls the optimizer's step size; weight decay shrinks weights to
regularize the model (`0` disables it). Both values appear in the training
output and the pipeline's saved run configuration.

Learning-rate scheduling defaults to `--lr-scheduler plateau`: after three
consecutive epochs without a new best validation loss (`--lr-patience 2`),
AdamW's rate is multiplied by `--lr-factor 0.5`, down to `--min-lr 0.000001`.
Each actual reduction resets early-stopping patience, giving the lower rate
time to work. Once no further reduction occurs, early stopping applies normally;
`--epochs` always caps training. Scheduler patience must be less than
early-stopping patience. Use `--lr-scheduler none` for a fixed learning rate.
The epoch log shows the rate used, reductions are printed, and `history.json`
records both `learning_rate` (used this epoch) and `next_learning_rate`.

After preparing the shared dataset and tokenizer, standalone training is also
available:

```powershell
uv run --locked python -m ghostwriter.training.train --model gru --epochs 3 --device cpu
```

This uses `data/processed/` and `data/tokenizer.json`, saving `best_gru.pt` and
its history in the repository root. Prefer the pipeline's isolated run folders
when comparing experiments so checkpoint/tokenizer pairs stay together.

Compare architectures by changing `--model` while keeping the dataset,
vocabulary, tokenization, context, and evaluation procedure fixed. Record
parameter counts, training time, training budget, and seeds. Use validation
results to choose settings and reserve test results for final assessment.

Evaluation reports cross-entropy, perplexity (`exp(loss)`), next-token
accuracy, and non-padding target count. Lower perplexity measures better
next-token prediction on that evaluation set, not necessarily better lyrics.
Final evaluation uses non-overlapping windows and counts each target once;
its scores can differ from overlapping-window metrics during training.

### Contrastive song embeddings

A separate experiment can train a projection head on a saved Transformer run.
It samples two non-overlapping lyric excerpts per training song, freezes the
Transformer, mean-pools non-padding hidden states, and trains the head with
same-song contrastive loss. The original Transformer checkpoint and tokenizer
remain the source for evaluation; contrastive artifacts go in a new ignored
`contrastive/<timestamp>/` folder within that run.

```powershell
python -m ghostwriter.training.train_contrastive --run-dir runs\20261011-100044-875583 --model transformer --device cpu
```

On the optional Windows AMD environment, use:

```powershell
.\gpu.ps1 -m ghostwriter.training.train_contrastive --run-dir runs\20261011-100044-875583 --model transformer --device cuda
```

The training command prints the actual output folder and follow-up commands.
`--epochs`, `--batch-size`, `--temperature`, `--learning-rate`,
`--weight-decay`, `--dropout`, `--seed`, and `--min-view-len` are configurable.
The saved `best_projection.pt` is selected by validation contrastive loss;
`history.json`, `metrics.json`, and `config.json` record the comparison.

```powershell
python -m ghostwriter.training.eval_contrastive --contrastive-dir runs\20261011-100044-875583\contrastive\YOUR_OUTPUT_FOLDER --split test
python -m ghostwriter.cluster_contrastive --contrastive-dir runs\20261011-100044-875583\contrastive\YOUR_OUTPUT_FOLDER --split test
```

Evaluation uses fixed sampled pairs and identical in-batch negatives for the
original pooled and projected vectors. It reports same-song top-1 retrieval
and contrastive loss. The chart command saves two interactive HTML charts,
one for the pooled Transformer embeddings and one for the projected
embeddings, with K-means clusters and artist labels parsed from filenames of
the form `artist__song.txt`. It also reports same-artist precision@5. Use
`--show` to open the charts. UMAP fits separately to each representation, so
compare neighborhoods and cluster membership rather than axis positions.
The test split is the default for both commands. Better same-song retrieval
does not by itself establish better style or semantic similarity.

```powershell
uv run --locked python -m ghostwriter.training.summarize --details
```

The summary lists saved settings, metrics, and reuse commands, and writes an
ignored `runs/summary.md`. Ranking is restricted to groups with matching
validation data, vocabulary, tokenizer settings, and evaluation context.
Word, character, and BPE perplexities use different prediction units and should
not be compared directly. Changing BPE vocabulary or merge rules also changes
the prediction units. Generated quality, repetition, and memorization need
additional inspection; one good sample does not establish generalization.

## 5. Evaluation and inference

Use the checkpoint and tokenizer from the same run. Replace the example run
name below with the directory printed by your GRU pipeline run:

```powershell
$run = "runs/REPLACE_WITH_RUN_NAME"
uv run --locked python -m ghostwriter.training.eval --checkpoint "$run/best_gru.pt" --tokenizer "$run/tokenizer.json" --data-dir "$run/processed" --split test --device cpu
uv run --locked python -m ghostwriter.inference.generate --checkpoint "$run/best_gru.pt" --tokenizer "$run/tokenizer.json" --device cpu
```

Standalone training instead uses `best_gru.pt`, `data/tokenizer.json`, and
`data/processed/`. For another architecture, use its `best_<model>.pt` file.
Loading recovers architecture and context settings and checks token identities
and tokenizer settings. Evaluation and generation reject context overrides
that differ from training.

Generation predicts a token, appends it, and feeds the latest context back
into the model until `<EOS>` or the new-token limit. Newlines become line
breaks; character tokens are joined directly, while word tokens receive
spacing and punctuation cleanup.

Generation normalizes excessive blank lines in the prompt, then rejects any
candidate token whose decoded continuation would create three consecutive
newlines, including whitespace-only blank lines. Rejected candidates are
masked before resampling, with top-k reapplied; greedy generation follows
the same constraint. BPE candidates are checked as decoded text, too. This
works with existing checkpoints; learning the cleaner spacing requires a
new training run with newly prepared data.

| Setting | Default or purpose |
| --- | --- |
| `--prompt` | `<INTRO>` with sections; empty without them; ordinary text is also accepted |
| `--max-new-tokens` | 200 for words/BPE, 1000 for characters |
| `--temperature` | 1.0; zero is greedy, larger positive values broaden sampling |
| `--top-k` | 40; limit sampling to top-scoring candidates, or use 0 for no restriction |
| `--seed` | Fresh randomness by default; supply a seed for repeatable sampling on the same setup |
| `--plain` | Print text without the terminal frame or colors |

## Local generation playground

Browse saved runs and generate text in a small local web app. The backend uses
[FastAPI](https://fastapi.tiangolo.com/) and the existing checkpoint loader and
generation function; the frontend is plain HTML, CSS, and JavaScript. No Node
installation, cloud service, or credentials are needed.

Start the standard CPU environment:

```sh
uv run --locked python -m ghostwriter.web
```

For the optional Windows AMD environment:

```powershell
.\gpu.ps1 -m ghostwriter.web
```

Open **http://127.0.0.1:8000**. Stop the server with Ctrl+C. It binds to
loopback for local use. Use `--port 8001` to change ports or
`--runs-dir runs/my-experiments` to browse another local runs directory.

Choose a run from the model picker in the top bar, enter an optional prompt,
and select Generate. Sampling controls live under Options; model files and
context are nested there. Cmd/Ctrl+Enter generates from the keyboard. Switching
models preserves a prompt you have written, and Reset defaults restores the
generation options without clearing it.

Each immediate run folder needs a `.pt` checkpoint and a saved tokenizer JSON.
The app exposes checkpoint/tokenizer selection, prompt, temperature, top-k,
new-token limit, seed, CPU/GPU device, and section-color/plain display options.
Context comes from the checkpoint; a manual context is only needed for legacy
weights missing their training window. The server checks tokenizer identity
and context with the same rules as the CLI.

Word/BPE generation defaults to 200 new tokens and character generation to
1000. Temperature defaults to 1.0, top-k to 40, and an empty seed creates fresh
randomness. The web app defaults to GPU when available, otherwise CPU; a
manual device choice survives Refresh. Run settings and validation metrics appear beside the controls.
Perplexities across different tokenizers are not directly comparable.

The app keeps up to ten samples in the current tab's memory, with the actual
seed, settings, and a command for reproducing each result in the server's
environment. Copy text or download it as UTF-8 `.txt`; page reloads clear the
sample history. Generating does not write to your runs or dataset. The server
caches one loaded model and runs one generation at a time. GPU execution is
available only when the server's PyTorch environment supports it.

For the API and existing project tests:

```sh
uv run --locked python -m unittest discover -s tests -v
```

The `dev` dependency group supplies the HTTP test client. Tests create
synthetic data and checkpoints in temporary directories.

## Optional AMD GPU execution

The optional environment in `environments/amd/` targets Windows x86-64,
Python 3.12, and AMD's `gfx1100` PyTorch build. It requires compatible hardware
and drivers and is separate from the standard CPU environment. It is not
required for the educational workflow. On a compatible setup, use:

```powershell
.\gpu.ps1 -m ghostwriter.pipeline --model gru --device cuda
```

The launcher installs its environment on first use. To select a device, set
`HIP_VISIBLE_DEVICES` in your shell. A local `.gpu-device` file containing a
device index can also supply a default; it is ignored by Git. Explicit shell
settings take precedence, and the launcher restores them afterward. ROCm
uses PyTorch's `cuda` device name. Generation and evaluation default to CPU;
pass `--device cuda` for GPU execution. Transformer attention uses the math
backend on Windows/ROCm for compatibility.

## Scope

The focus is understanding data splits, tokenization, embeddings, tensor
shapes, sequence architectures, optimization, evaluation, and autoregressive
sampling. Tiny datasets can overfit quickly, and generated text can repeat
training material. The synthetic examples demonstrate mechanics only; they
are not a benchmark for model quality or a substitute for a carefully
prepared, permitted dataset.
