"""Response-only GPT-2 LoRA fine-tuning with purpose-grouped held-out splits."""
import argparse
import hashlib
import json
import math
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

MODEL_ID = "openai-community/gpt2"
DATASET_ID = "Kamisori-daijin/email-datasets-20k"
DATA_REVISION = "7c615780b64770697e8e517c9d3e70db470cadf1"
DATA_URL = (f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATA_REVISION}/"
            "email-datasets-20k-v2.jsonl")


def format_prompt(instruction):
    return f"### Request:\n{instruction.strip()}\n\n### Email:\n"


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def purpose_group(instruction):
    # The source varies roles, tones and situations around an email purpose.
    match = re.search(r"regarding (.*?)(?:, specifically|\.$|$)", instruction, re.I)
    return normalize(match.group(1) if match else instruction)


def clean_and_split(rows, seed=42):
    cleaned, seen = [], set()
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        try:
            email = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]
            subject, body = email["subject"].strip(), email["body"].strip()
            # Some source rows contain doubly escaped newlines/Unicode.
            body = body.replace("\\n", "\n").replace("\\t", "\t")
            body = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), body)
            if body.lower().startswith("subject:"):
                body = body.split("\n", 1)[-1].lstrip()
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if not instruction or not subject or len(body) < 80:
            continue
        response = f"Subject: {subject}\n\n{body}"
        key = normalize(response)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"id": row.get("id"), "instruction": instruction,
                        "response": response, "group": purpose_group(instruction)})
    groups = sorted({r["group"] for r in cleaned})
    if len(groups) < 10:
        raise ValueError("At least ten distinct email-purpose groups required.")
    rng = random.Random(seed)
    rng.shuffle(groups)
    count = max(1, round(len(groups) * 0.1))
    test, val = set(groups[:count]), set(groups[count:2 * count])
    splits = {"train": [], "val": [], "test": []}
    for row in cleaned:
        name = "test" if row["group"] in test else "val" if row["group"] in val else "train"
        splits[name].append(row)
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def encode_record(row, tokenizer, max_length):
    prompt = tokenizer.encode(format_prompt(row["instruction"]), add_special_tokens=False)
    response = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Skip overlength records; do not teach the model to end an incomplete email.
    if len(prompt) + len(response) > max_length:
        return None
    return {"input_ids": prompt + response, "labels": [-100] * len(prompt) + response}


def collate(records, pad_id):
    length = max(len(r["input_ids"]) for r in records)
    return {"input_ids": torch.tensor([r["input_ids"] + [pad_id] * (length - len(r["input_ids"]))
                                        for r in records]),
            "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"]))
                                             for r in records]),
            "labels": torch.tensor([r["labels"] + [-100] * (length - len(r["labels"]))
                                     for r in records])}


@torch.inference_mode()
def evaluate(model, loader, device, autocast):
    model.eval()
    total, tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        count = (batch["labels"][:, 1:] != -100).sum().item()
        with autocast():
            loss = model(**batch).loss
        total += loss.item() * count
        tokens += count
    if tokens == 0:
        raise ValueError("No supervised evaluation tokens.")
    loss = total / tokens
    return {"response_loss": loss, "response_perplexity": math.exp(loss), "tokens": tokens}


@torch.inference_mode()
def generate_email(model, tokenizer, instruction, device, max_new_tokens=250):
    model.eval()
    inputs = tokenizer(format_prompt(instruction), return_tensors="pt").to(device)
    if inputs.input_ids.shape[1] + max_new_tokens > model.config.n_positions:
        raise ValueError("Prompt plus requested output exceeds GPT-2 context.")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id, use_cache=True)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/email-datasets-20k-v2.jsonl")
    parser.add_argument("--output", default="outputs/email-lora")
    parser.add_argument("--cache-dir", default="checkpoints")
    parser.add_argument("--train-examples", type=int, default=5000)
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume-adapter", help="Continue adapter weights with a fresh optimizer")
    args = parser.parse_args()
    if min(args.train_examples, args.eval_examples, args.epochs, args.batch_size,
           args.accumulation, args.threads) < 1 or not 64 <= args.max_length <= 1024:
        raise ValueError("Positive counts required; max_length must be in [64, 1024].")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU BF16 accelerates supported hosts; keep CPU portable by using FP32 by default.
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" and torch.cuda.is_bf16_supported() else nullcontext
    out, data = Path(args.output), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    if not data.exists():
        import requests
        data.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(DATA_URL, timeout=120)
        response.raise_for_status()
        data.write_bytes(response.content)
    raw_bytes = data.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.decode().splitlines() if line.strip()]
    splits = clean_and_split(rows, args.seed)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    datasets, retained = {}, {}
    for name, rows in splits.items():
        limit = args.train_examples if name == "train" else args.eval_examples
        encoded, selected = [], []
        for row in rows:
            record = encode_record(row, tokenizer, args.max_length)
            if record is not None:
                encoded.append(record)
                selected.append(row)
            if len(encoded) == limit:
                break
        if not encoded:
            raise ValueError(f"Empty {name} split after length filtering.")
        datasets[name], retained[name] = encoded, selected
    loader_args = {"batch_size": args.batch_size, "collate_fn": lambda x: collate(x, tokenizer.pad_token_id)}
    loaders = {name: DataLoader(ds, shuffle=(name == "train"), **loader_args)
               for name, ds in datasets.items()}
    base = GPT2LMHeadModel.from_pretrained(MODEL_ID, cache_dir=args.cache_dir,
                                          use_safetensors=True).to(device)
    base.config.use_cache = False
    print("Device:", device, "Records:", {n: len(v) for n, v in datasets.items()}, flush=True)
    baseline_val = evaluate(base, loaders["val"], device, autocast)
    baseline_test = evaluate(base, loaders["test"], device, autocast)
    sample_rows = retained["test"][:3]
    samples = [{"instruction": r["instruction"], "reference": r["response"],
                "before": generate_email(base, tokenizer, r["instruction"], device)} for r in sample_rows]
    if args.resume_adapter:
        model = PeftModel.from_pretrained(base, args.resume_adapter, is_trainable=True)
    else:
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                            lora_dropout=0.05, target_modules=["c_attn"],
                            fan_in_fan_out=True, bias="none")
        model = get_peft_model(base, config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.learning_rate, weight_decay=0.01)
    best, history, step = float("inf"), [], 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loaders["train"]):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Normalize the final incomplete accumulation window correctly.
            group_start = (index // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, len(loaders["train"]) - group_start)
            with autocast():
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            (loss / group_size).backward()
            if (index + 1) % args.accumulation == 0 or index + 1 == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(f"Epoch {epoch+1} step {step} loss {loss.item():.4f} elapsed {time.perf_counter()-started:.1f}s", flush=True)
        val = evaluate(model, loaders["val"], device, autocast)
        history.append({"epoch": epoch + 1, "step": step, **val})
        if val["response_loss"] < best:
            best = val["response_loss"]
            model.save_pretrained(out / "best", safe_serialization=True)
            tokenizer.save_pretrained(out / "best")
        model.save_pretrained(out / "last", safe_serialization=True)
        tokenizer.save_pretrained(out / "last")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    model.load_adapter(out / "best", adapter_name="best_eval")
    model.set_adapter("best_eval")
    final_test = evaluate(model, loaders["test"], device, autocast)
    for row in samples:
        row["after"] = generate_email(model, tokenizer, row["instruction"], device)
    results = {"base_model": MODEL_ID, "dataset": DATASET_ID, "dataset_revision": DATA_REVISION,
               "data_sha256": hashlib.sha256(raw_bytes).hexdigest(),
               "device": device, "config": vars(args), "optimizer_steps": step,
               "examples": {n: len(v) for n, v in datasets.items()},
               "groups": {n: sorted({r["group"] for r in rows}) for n, rows in splits.items()},
               "baseline_validation": baseline_val, "baseline_test": baseline_test,
               "best_validation_loss": best, "finetuned_test": final_test,
               "history": history, "training_seconds": time.perf_counter() - started,
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "samples.json").write_text(json.dumps(samples, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
