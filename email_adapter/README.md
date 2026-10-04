---
language: en
base_model: openai-community/gpt2
library_name: peft
pipeline_tag: text-generation
datasets:
- Kamisori-daijin/email-datasets-20k
tags:
- lora
- email-writing
- experimental
---

# MiniGPT email-writing LoRA — initial checkpoint

Actual trained LoRA adapter for OpenAI GPT-2 small. It requires the unchanged GPT-2 base weights and tokenizer; this folder is not a standalone language model.

## Training

- Dataset: Kamisori-daijin/email-datasets-20k, revision 7c615780b64770697e8e517c9d3e70db470cadf1.
- 512 cleaned, length-filtered training emails; 2 epochs; 128 optimizer steps.
- Rank 8, alpha 16, dropout 0.05, target c_attn; 294,912 trainable parameters.
- Response-only loss; prompt/padding labels ignored.
- CPU, FP32, seed 42; maximum sequence length 384.
- Email-purpose-grouped splits; 32 validation and 32 test examples.
- Best adapter selected by validation response loss.

## Measured results

| Metric | Base GPT-2 | Fine-tuned adapter |
| --- | --- | --- |
| Test response cross-entropy | 3.3610 | 2.5126 |
| Test response perplexity | 28.82 | 12.34 |

This is a small synthetic-data evaluation, not a writing-quality benchmark. See ../email_training_results for raw metrics, held-out samples, and a manually written request.

## Use

From the repository root:

```bash
python -m pip install -r requirements.txt
python write_email.py --adapter email_adapter --request "Write a polite email requesting a project meeting."
```

The script formats the prompt as `### Request:` followed by `### Email:` and loads this adapter with PEFT. Base weights download on first use.

## Observed limitations

Generation became more email-like but still invents details, mixes sender/recipient roles, produces stray markup, and sometimes ignores the request. A fresh professor-email test invented research problems and a phone number. The checkpoint is experimental and is not a reliable email assistant. Inspect every draft; do not treat supplied names, dates, contact information or claims as verified.

Training uses business-email examples, including unusual tones and situations. It is not comprehensive academic-email or letter-writing training. Larger curated instruction-response training and a manually reviewed evaluation set are appropriate next experiments. Low test perplexity does not establish reliable instruction following.

## Sources and usage terms

Base model: https://huggingface.co/openai-community/gpt2

Dataset: https://huggingface.co/datasets/Kamisori-daijin/email-datasets-20k

The dataset card lists Apache 2.0 and refers to Gemma usage terms for generated content. Follow the original dataset and model terms; this card does not replace them.
