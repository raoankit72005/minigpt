"""GPT-2-compatible PyTorch model and explicit pretrained weight mapping.

Reference: rasbt/LLMs-from-scratch, ch05/02_alternative_weight_loading.
Transformers is used to download/read weights and tokenize, not for our forward pass.
"""
import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "openai-community/gpt2"


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    context_length: int = 1024
    embedding_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    layer_norm_epsilon: float = 1e-5


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.num_heads
        self.head_dim = cfg.embedding_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embedding_dim, 3 * cfg.embedding_dim)
        self.projection = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for a in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        result = F.softmax(scores, dim=-1) @ v
        return self.projection(result.transpose(1, 2).contiguous().reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embedding_dim, eps=cfg.layer_norm_epsilon)
        self.attention = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embedding_dim, eps=cfg.layer_norm_epsilon)
        self.mlp = nn.Sequential(nn.Linear(cfg.embedding_dim, 4 * cfg.embedding_dim),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * cfg.embedding_dim, cfg.embedding_dim))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT2(nn.Module):
    """Single-sequence/unpadded-batch inference model; no KV cache or dropout."""
    def __init__(self, cfg):
        super().__init__()
        if min(cfg.vocab_size, cfg.context_length, cfg.embedding_dim,
               cfg.num_heads, cfg.num_layers) < 1 or cfg.embedding_dim % cfg.num_heads:
            raise ValueError("Positive dimensions required; embedding width must divide by heads.")
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.embedding_dim, eps=cfg.layer_norm_epsilon)
        self.output = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.cfg.context_length:
            raise ValueError("Expected (batch, length) with length within context_length.")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens=100, temperature=0.8, top_k=40, eos_id=50256):
        if max_new_tokens < 0 or temperature <= 0 or top_k < 0:
            raise ValueError("Nonnegative token count/top_k and positive temperature required.")
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Generation expects one nonempty prompt.")
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg.context_length:])[:, -1, :] / temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat((ids, next_id), dim=1)
            if next_id.item() == eos_id:
                break
        return ids


def config_from_hf(cfg):
    if cfg.activation_function != "gelu_new" or not cfg.scale_attn_weights:
        raise ValueError("Expected standard GPT-2 activation and attention scaling.")
    if cfg.scale_attn_by_inverse_layer_idx or cfg.reorder_and_upcast_attn:
        raise ValueError("Nonstandard GPT-2 attention options are unsupported.")
    return GPT2Config(cfg.vocab_size, cfg.n_positions, cfg.n_embd, cfg.n_head,
                      cfg.n_layer, cfg.layer_norm_epsilon)


@torch.no_grad()
def load_hf_weights(model, state):
    """Copy every parameter, transposing GPT-2 Conv1D weights into nn.Linear layout."""
    mapped = {
        "token_embedding.weight": state["transformer.wte.weight"],
        "position_embedding.weight": state["transformer.wpe.weight"],
        "final_norm.weight": state["transformer.ln_f.weight"],
        "final_norm.bias": state["transformer.ln_f.bias"],
        "output.weight": state["lm_head.weight"],
    }
    if not torch.equal(mapped["token_embedding.weight"], mapped["output.weight"]):
        raise ValueError("Expected GPT-2 tied token embedding/output weights.")
    for i in range(model.cfg.num_layers):
        source, target = f"transformer.h.{i}", f"blocks.{i}"
        for ours, theirs in [("norm1", "ln_1"), ("norm2", "ln_2"),
                             ("attention.qkv", "attn.c_attn"),
                             ("attention.projection", "attn.c_proj"),
                             ("mlp.0", "mlp.c_fc"), ("mlp.2", "mlp.c_proj")]:
            for suffix in ("weight", "bias"):
                value = state[f"{source}.{theirs}.{suffix}"]
                if suffix == "weight" and theirs not in ("ln_1", "ln_2"):
                    value = value.T
                mapped[f"{target}.{ours}.{suffix}"] = value
    expected = model.state_dict()
    if set(expected) != set(mapped):
        raise ValueError("Incomplete parameter mapping.")
    for key, value in mapped.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"Shape mismatch for {key}: {value.shape} vs {expected[key].shape}")
    model.load_state_dict(mapped, strict=True)
    return model.eval()


def load_pretrained(cache_dir="checkpoints", verify=True):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    reference = GPT2LMHeadModel.from_pretrained(
        MODEL_ID, cache_dir=cache_dir, use_safetensors=True, attn_implementation="eager"
    ).eval()
    model = load_hf_weights(GPT2(config_from_hf(reference.config)), reference.state_dict())
    if verify:
        with torch.inference_mode():
            ids = torch.tensor([[15496, 11, 995, 0]])
            torch.testing.assert_close(model(ids), reference(ids).logits, rtol=1e-4, atol=1e-4)
        print("PASS: mapped pretrained logits match the reference model.")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text using OpenAI GPT-2 weights.")
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cache-dir", default="checkpoints")
    parser.add_argument("--save", help="Optional local path for the converted checkpoint")
    args = parser.parse_args()
    model, tokenizer = load_pretrained(args.cache_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = model.to(device)
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(model.cfg), "model_id": MODEL_ID,
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, path)
    ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    if ids.shape[1] == 0:
        ids = torch.tensor([[tokenizer.eos_token_id]], device=device)
    torch.manual_seed(42)
    result = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k,
                            tokenizer.eos_token_id)
    print(tokenizer.decode(result[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
