"""Checkpoint loading and greedy generation for CD-SSM."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from cdssm.config import CDSSMConfig
from cdssm.models.model import CDSSMLMHeadModel

_COMPUTED_FIELDS = {"d_inner", "n_heads", "head_dim", "ssm_norm_groups"}


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    init_kwargs = {
        k: v for k, v in cfg_dict.items()
        if k in CDSSMConfig.__dataclass_fields__ and k not in _COMPUTED_FIELDS
    }
    config = CDSSMConfig(**init_kwargs)
    model = CDSSMLMHeadModel(config, cfg_dict.get("vocab_size", 50257))

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).bfloat16().eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 64, device: str = "cuda") -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        next_token = logits[0, -1].argmax().item()
        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token]], device=device, dtype=torch.long),
        ], dim=1)

    return tokenizer.decode(input_ids[0].tolist())
