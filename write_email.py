"""Generate an email with the trained LoRA adapter or export it to custom GPT-2."""
import argparse
from pathlib import Path
from dataclasses import asdict
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from peft import PeftModel
from finetune_email import MODEL_ID, generate_email


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="email_adapter")
    parser.add_argument("--request", default="Write a professional business email requesting a meeting to discuss a project update.")
    parser.add_argument("--cache-dir", default="checkpoints")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--export-custom", help="Merge LoRA weights and save a checkpoint for custom GPT2")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = GPT2LMHeadModel.from_pretrained(MODEL_ID, cache_dir=args.cache_dir, use_safetensors=True)
    model = PeftModel.from_pretrained(base, args.adapter).eval().to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print(generate_email(model, tokenizer, args.request, device, args.max_new_tokens))
    if args.export_custom:
        from gpt2_pretrained import GPT2, config_from_hf, load_hf_weights
        merged = model.merge_and_unload().cpu().eval()
        custom = load_hf_weights(GPT2(config_from_hf(merged.config)), merged.state_dict())
        with torch.inference_mode():
            ids = tokenizer.encode("### Request:\nWrite an email.\n\n### Email:\n", return_tensors="pt")
            torch.testing.assert_close(custom(ids), merged(ids).logits, rtol=1e-4, atol=1e-4)
        path = Path(args.export_custom)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(custom.cfg), "model_id": MODEL_ID,
                    "state_dict": custom.state_dict()}, path)
        print("Saved merged custom checkpoint:", path)


if __name__ == "__main__":
    main()
