"""Gradio portfolio demo for the included GPT-2 email adapter."""
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_ID = "openai-community/gpt2"


@lru_cache(maxsize=1)
def load_model():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = os.environ.get("MINIGPT_CACHE_DIR", "checkpoints")
    base = GPT2LMHeadModel.from_pretrained(MODEL_ID, cache_dir=cache, use_safetensors=True)
    adapter = Path(__file__).resolve().parent / "email_adapter"
    model = PeftModel.from_pretrained(base, str(adapter)).eval().to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID, cache_dir=cache)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


@torch.inference_mode()
def draft_email(request, max_new_tokens=160):
    request = (request or "").strip()
    if not request:
        raise gr.Error("Describe the email you want to write.")
    if len(request) > 3000:
        raise gr.Error("Please shorten your request to 3,000 characters.")
    count = int(max_new_tokens)
    if not 50 <= count <= 250:
        raise gr.Error("Choose an output length between 50 and 250 tokens.")
    model, tokenizer, device = load_model()
    prompt = f"### Request:\n{request}\n\n### Email:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    length = inputs.input_ids.shape[1]
    if length + count > model.config.n_positions:
        raise gr.Error("Your request is too long. Shorten it or reduce the output length.")
    output = model.generate(**inputs, max_new_tokens=count, do_sample=False,
                            repetition_penalty=1.1, use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()


def build_demo():
    with gr.Blocks(title="MiniGPT Email Writer") as demo:
        gr.Markdown("# MiniGPT Email Writer\nDraft a business email with GPT-2 and a trained LoRA adapter.")
        gr.Markdown("**Experimental demo:** drafts can invent details or miss instructions. Review names, dates, and claims before using them.")
        request = gr.Textbox(label="Describe your email", lines=4,
                             placeholder="Write a polite email requesting a project meeting next week.")
        length = gr.Slider(50, 250, value=160, step=10, label="Maximum output tokens")
        button = gr.Button("Generate draft", variant="primary")
        output = gr.Textbox(label="Email draft", lines=14, interactive=False)
        button.click(draft_email, [request, length], output, api_name="draft_email")
        gr.Examples([
            ["Write a professional email requesting a project status update."],
            ["Write a polite follow-up email after a business meeting."],
            ["Write a business email requesting a product demonstration."],
        ], inputs=request)
        gr.Markdown("Built by Ankit Yadav · [Source and training results](https://github.com/raoankit72005/minigpt)\n\nGPT-2 (124M) · rank-8 LoRA · initial training: 512 synthetic emails, two epochs.")
    return demo.queue(max_size=10, default_concurrency_limit=1, api_open=False)


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
