# MiniGPT — From Scratch and Pretrained GPT-2

This repository contains three workflows: a small character-level GPT trained from scratch, a custom GPT-2 model that loads OpenAI's released weights, and LoRA fine-tuning for business-email writing.

## Email-writing fine-tuning

[![Fine-tune emails in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raoankit72005/minigpt/blob/main/GPT2_Email_Finetuning_Colab.ipynb)

This workflow uses GPT-2 with **response-only LoRA fine-tuning** on synthetic business-email pairs. Training uses the Transformers GPT-2 implementation, which supplies the training behavior and dropout omitted from the custom inference model. The trained adapter can also be merged and mapped back into the custom PyTorch model.

### Initial training run completed

The included adapter was actually trained on **512 emails for two epochs (128 optimizer updates)** on CPU, with rank-8 LoRA and 294,912 trainable parameters.

| Held-out metric | Base GPT-2 | Fine-tuned |
| --- | --- | --- |
| Test response loss | 3.3610 | 2.5126 |
| Test response perplexity | 28.82 | 12.34 |
| Validation response perplexity | 26.32 | 12.67 |

Validation and test each contain 32 emails drawn from separate held-out purpose groups. These small, synthetic-data measurements demonstrate improved dataset likelihood, not reliable writing quality.

**Observed limitations:** outputs became more email-like, but still invented facts, mixed sender/recipient roles, and included stray markup. A fresh professor-email request produced an invented phone number. This is an experimental checkpoint, not a finished email assistant. The repository includes the actual outputs so the improvement and remaining problems are reviewable.

### Use the included trained adapter

```bash
python -m pip install -r requirements.txt
python write_email.py --adapter email_adapter --request "Write a polite email requesting a project meeting next week."
```

The small LoRA adapter is included in `email_adapter/`; the original GPT-2 base weights download on first use. The adapter alone is not a complete model. Generation uses the same Request/Email format as training and produces an email draft for review.

### Train a larger run

```bash
python finetune_email.py --train-examples 5000 --eval-examples 200 --epochs 2 --output outputs/email-lora
python write_email.py --adapter outputs/email-lora/best --request "Write a formal email asking for a project status update."
```

Use the new Colab notebook for GPU training. It includes saving to Google Drive, measured baseline/fine-tuned comparisons, inference, and adapter download.

To continue from the included adapter, add `--resume-adapter email_adapter`. This loads its weights and starts a **fresh optimizer**; it does not reproduce uninterrupted training. The `best/` adapter is selected using validation response loss; `last/` retains the final epoch's weights.

### Data and evaluation

Dataset: [Kamisori-daijin/email-datasets-20k](https://huggingface.co/datasets/Kamisori-daijin/email-datasets-20k), pinned to revision `7c615780b64770697e8e517c9d3e70db470cadf1`. Its card lists Apache 2.0 and asks users to refer to the Gemma usage terms. The dataset contains synthetic business emails generated with Gemma; it does not provide broad coverage of academic applications or formal letters.

Preprocessing parses subject/body, repairs escaped newlines, removes duplicate responses, and skips malformed/overlength records. All examples for the same parsed email purpose stay together in one split. This reduces template leakage, but is not a comprehensive semantic near-duplicate audit. Prompt and padding labels are masked with `-100`; loss is computed on response tokens only.

Metrics are token-weighted held-out **response** cross-entropy and perplexity. They measure likelihood on this dataset, not factual accuracy, suitability, or comprehensive writing quality. Read `email_training_results/samples.json` for actual before/after outputs and test fresh requests yourself.

### Export merged weights for the custom model

```bash
python write_email.py --adapter email_adapter --export-custom outputs/email-gpt2.pt
```

The export merges the LoRA adapter into GPT-2, maps it into the custom model, and checks logits against the merged reference. The resulting checkpoint is roughly 500 MB and should stay outside Git.

### Files

| File | Purpose |
| --- | --- |
| `finetune_email.py` | Dataset preparation, grouped splits, masked-loss LoRA training, checkpoint selection, and evaluation |
| `write_email.py` | Trained-adapter inference and optional custom-model export |
| `GPT2_Email_Finetuning_Colab.ipynb` | GPU training and inference notebook |
| `email_adapter/` | Initial trained adapter and its model card |
| `email_training_results/` | Measured results and before/after samples |
| `test_email_pipeline.py` | Response-mask, padding, and split-isolation tests |

Run checks with `python -m unittest -v test_email_pipeline.py test_gpt2_pretrained.py`.


## Use pretrained GPT-2 (124M)

[![Open pretrained GPT-2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raoankit72005/minigpt/blob/main/GPT2_Pretrained_Colab.ipynb)

The pretrained workflow uses `openai-community/gpt2`, the GPT-2 BPE tokenizer, 12 Transformer layers, 12 attention heads, 768-dimensional embeddings, and a 1,024-token context. It runs inference through our explicit PyTorch implementation rather than a Transformers forward pass.

Your original character-level model has a different vocabulary and tensor shapes, so GPT-2 weights cannot be loaded into it. Its training notebook remains available below.

### Local setup and generation

Use Python 3.10 or newer in a virtual environment, clone this repository, and run:

```bash
python -m pip install -r requirements.txt
python gpt2_pretrained.py --prompt "Every effort moves you" --max-new-tokens 100
```

The first run downloads about 500 MB of model weights plus tokenizer files into `checkpoints/`. Internet access is needed for that download. No paid inference API or API key is required. CPU is supported; CUDA is selected automatically when available. Loading temporarily holds both the reference and custom model in CPU memory, so allow several GB of free RAM.

The loader copies all pretrained parameters with shape checks, transposes GPT-2's Conv1D weights into PyTorch Linear layout, ties output and token embeddings, and verifies logits against the downloaded reference model before generation. The custom model uses the GPT-2 tanh GELU approximation and LayerNorm epsilon.

```bash
# Force CPU and save a converted local checkpoint
python gpt2_pretrained.py --device cpu --save checkpoints/gpt2.pt

# Run offline mapping, causal masking, and generation checks
python -m unittest -v test_gpt2_pretrained.py
```

The Colab notebook includes saving/restoring a converted checkpoint and tokenizer. Large model files are downloaded at runtime and ignored by Git; they are not committed to this repository. The CLI loads the original Hugging Face checkpoint from the cache on subsequent runs; `--save` exports a custom checkpoint, not a CLI resume option.

GPT-2 produces text continuations. It is not instruction-tuned, and loading these weights does not turn it into a ChatGPT-style assistant. This implementation focuses on inference with unpadded inputs; it has no KV cache and recomputes the active context for each generated token.

### Pretrained workflow files

| File | Purpose |
| --- | --- |
| `gpt2_pretrained.py` | GPT-2 architecture, explicit weight mapping, parity check, and generation CLI |
| `GPT2_Pretrained_Colab.ipynb` | Download, generate, save, and restore in Colab |
| `test_gpt2_pretrained.py` | Offline tests against a small randomly initialized Transformers GPT-2 |
| `requirements.txt` | Dependencies for all workflows |

### Validation

The four offline tests passed: logits match the Transformers reference at multiple sequence lengths, causal masking prevents future-token leakage, incompatible tensor shapes are rejected, and generation handles the context boundary.

The actual GPT-2 small safetensors checkpoint was downloaded and loaded during validation. Its mapped logits passed the reference comparison. A CPU generation smoke test with prompt `Machine learning is`, seed 42, temperature 0.8, top-k 40, and 10 new tokens produced:

> Machine learning is more than just a tool. It has more than

This is a generation smoke test, not a task-quality benchmark.

### Reference and attribution

The weight-loading approach follows [Sebastian Raschka's LLMs-from-scratch, Chapter 5 alternative weight loading](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/02_alternative_weight_loading): copy embeddings and normalization parameters, transpose projection weights, and preserve GPT-2's output embeddings.

Weights and model documentation: [OpenAI GPT-2 on Hugging Face](https://huggingface.co/openai-community/gpt2). Respect the model and dependency licenses. This is a pretrained-weight integration, not a claim that this project trained GPT-2.

---

# Original experiment — Character-Level Language Model from Scratch

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
