# Modeling notes

Ghostwriter treats writing as next-token prediction: given preceding tokens,
predict the token that follows. The same data pipeline feeds each architecture.

## Text to sequences

Songs are split before windows are created. The vocabulary is fitted on the
training songs; validation and test songs use that vocabulary unchanged.
Tokenization supports words, characters, or locally trained byte-level BPE.
Special tokens mark padding, unknown words, song boundaries, newlines, and
optional section labels.

Inputs and targets are shifted by one token. For example, `[BOS, a, b]`
predicts `[a, b, c]`. Windows stay within a song, and padding targets are
excluded from loss and accuracy.

## Models and shapes

```text
Token IDs          [batch, sequence]
Embeddings         [batch, sequence, embedding width]
Hidden states      [batch, sequence, hidden width]
Vocabulary logits  [batch, sequence, vocabulary size]
```

| Model | How it processes context |
| --- | --- |
| RNN | Updates a hidden state one token at a time. |
| LSTM | Uses gates and a separate cell state to control retained information. |
| GRU | Uses a gated recurrent state without a separate cell state. |
| Transformer | Combines token and positional embeddings with causal self-attention. |

Recurrent states reset between windows. The Transformer has two layers and
four attention heads; its causal mask blocks attention to future tokens.
A linear head maps each position's representation to vocabulary logits.

## Training and generation

Cross-entropy measures the negative log probability assigned to each target
token. Perplexity is `exp(mean token loss)`; token accuracy is the fraction of
correct highest-logit predictions. Scores depend on the tokenizer and context.

Training supplies the actual preceding tokens (teacher forcing). Generation
instead appends the model's own predictions and feeds the latest context back
into the model. Temperature and top-k control sampling.

The training loop uses AdamW, gradient clipping, dropout, and validation-based
early stopping. [Experiments](EXPERIMENTS.md) reports the recorded comparisons.
