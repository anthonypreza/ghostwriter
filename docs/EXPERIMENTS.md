# Experiments

These experiments compared recurrent models and a causal Transformer on
next-token prediction, then tested whether a longer context helped the LSTM.

## Setup

The October 10, 2026 runs used 1,039 songs, split by song into 831 training,
104 validation, and 104 test examples before sequence windows were created.
The word-level vocabulary contained 5,693 tokens and was built from training
text. All four runs used seed 42, batch size 32, stride 16, and dropout 0.1.
Training ran on CPU with a 30-epoch limit and early-stopping patience of five.

The RNN and LSTM used 64-dimensional embeddings and 128 hidden units. The
Transformer used 128-dimensional embeddings, two attention layers, and four
heads. Model sizes were therefore different.

## Results

| Model | Context length | Validation perplexity | Validation token accuracy | Test perplexity |
| --- | ---: | ---: | ---: | ---: |
| RNN | 32 | 82.37 | 24.14% | 80.30 |
| LSTM | 32 | 79.05 | 24.37% | 78.93 |
| Transformer | 32 | 79.33 | 24.66% | 77.11 |
| LSTM | 64 | 79.74 | 24.67% | 77.83 |

Lower perplexity means better next-token prediction on this evaluation set.
Token accuracy measures how often the highest-scoring token matched the target.
Final evaluation counted 36,611 validation targets and 35,836 test targets,
using non-overlapping windows.

## Observations

- At context length 32, the LSTM and Transformer had lower validation
  perplexity than the vanilla RNN. Their validation accuracies were close:
  24.37% and 24.66%.
- The LSTM had slightly lower validation perplexity than the Transformer,
  while the Transformer had lower test perplexity. Neither was consistently
  ahead across these metrics.
- Increasing the LSTM context from 32 to 64 did not improve validation
  perplexity, although its test perplexity decreased from 78.93 to 77.83.

These are single-seed observations with different model sizes, rather than
evidence of a general architecture ranking. The metrics describe token
prediction; they do not measure the quality or originality of generated text.
