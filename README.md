# MiniGPT — Character-Level Language Model from Scratch

A self-contained PyTorch experiment that trains a small decoder-only Transformer from random initialization on **Tiny Shakespeare**. The notebook implements tokenization, causal attention, training, text generation, checkpointing, and held-out evaluation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raoankit72005/minigpt/blob/main/GPT_Colab_Training.ipynb)

## What this project implements

- A character tokenizer with a vocabulary fitted on training text.
- Multi-head causal self-attention with an explicit attention mask.
- Learned token and position embeddings.
- Pre-normalized Transformer blocks with residual connections and GELU feed-forward layers.
- Next-character prediction using cross-entropy loss.
- AdamW optimization, gradient clipping, and validation-based checkpoint selection.
- Temperature and top-k sampling for autoregressive text generation.
- Learning curves, before/after text comparisons, and test metrics.

The model learns character patterns and Shakespeare-like text. It is not instruction-tuned and is not designed to answer questions as a chatbot.

## Default experiment

| Setting | Value |
| --- | --- |
| Dataset | Tiny Shakespeare |
| Split | First 90% training, next 5% validation, final 5% test |
| Tokenization | Character-level; vocabulary fitted on training text |
| Context length | 128 characters |
| Embedding dimension | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Gradient clipping | Maximum norm 1.0 |
| Initial training budget | 1,000 optimizer steps |
| Evaluation interval | 100 steps |

The default configuration has approximately **0.8 million parameters**; the notebook prints the exact count.

## Run in Google Colab

1. Open the notebook using the **Open in Colab** button above.
2. Choose **Runtime → Change runtime type → GPU**.
3. Run the cells in order. The notebook checks CUDA availability and downloads the dataset automatically.
4. Inspect the untrained baseline, training/validation curves, and generated samples.
5. Run the final evaluation after selecting settings using validation.
6. Download the results ZIP before the runtime ends.

The notebook uses PyTorch and Matplotlib, typically available in Colab. It requires internet access to download the dataset and does not require a hosted LLM API or API key. As written, it requires a CUDA GPU and uses Colab-specific storage and download paths.

## Continue training

The first training run performs **1,000 optimizer steps**, not 1,000 epochs.

To continue the same in-memory model, change `ADDITIONAL_STEPS` in the training cell—for example, to `4000`—and rerun **only that cell**, followed by the results cells. Rerunning the configuration cell resets the model and optimizer.

If GPU memory runs out, reduce `BATCH_SIZE` to 16. Training time depends on the assigned GPU and experiment settings.

## Checkpoints and generated artifacts

By default, outputs are saved under `/content/gpt_results`. Set `SAVE_TO_DRIVE = True` before training to use the notebook's Google Drive save option.

| Artifact | Contents |
| --- | --- |
| `best.pt` | Checkpoint with the lowest observed validation loss |
| `last.pt` | Checkpoint from the most recent evaluation |
| `history.json` | Training and validation loss history |
| `loss_curve.png` | Learning curve plot |
| `before.txt` | Text generated before training |
| `samples.json` | Generated samples at temperatures 0.5, 0.8, and 1.0 |
| `results.json` | Final metrics and model configuration |

Checkpoints contain the model configuration, character vocabulary, weights, optimizer state, step count, and evaluation losses. The notebook includes optional restoration instructions for a disconnected session; restoration does not reproduce the exact previous random-number stream.

## Evaluation

Validation loss is estimated on fixed sampled windows and selects the best checkpoint. Final test loss is estimated over 100 sampled batches of held-out text.

The notebook reports:

- Cross-entropy in nats per character.
- Character-level perplexity: `exp(test_loss)`.
- Bits per character: `test_loss / ln(2)`.

**No measured training results are embedded in the committed notebook.** Run the experiment to obtain metrics and samples. Character-level perplexity should not be compared directly with word- or subword-level perplexity. Keep the test set separate from repeated model tuning.

## Repository files

| File | Purpose |
| --- | --- |
| `GPT_Colab_Training.ipynb` | Complete model, training, generation, and evaluation workflow |
| `README.md` | Project overview and usage instructions |

Trained checkpoints and dataset files are generated during execution and are not included in the repository.

## Dataset source

[Tiny Shakespeare in Andrej Karpathy's char-rnn repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)

## Author

[Ankit Yadav](https://github.com/raoankit72005)
