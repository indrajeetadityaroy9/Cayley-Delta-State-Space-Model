"""Checkpoint loading and greedy generation for CD-SSM."""

from __future__ import annotations

import warnings

import torch
from transformers import AutoTokenizer

from cdssm.config.defaults import CDSSMConfig, COMPUTED_FIELDS
from cdssm.models.model import CDSSMLMHeadModel

_DEVICE = torch.device("cuda")


def load_model_from_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    init_kwargs = {
        k: v for k, v in cfg_dict.items()
        if k in CDSSMConfig.__dataclass_fields__ and k not in COMPUTED_FIELDS
    }
    config = CDSSMConfig(**init_kwargs)
    model = CDSSMLMHeadModel(config)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    result = model.load_state_dict(cleaned, strict=False)
    if result.missing_keys:
        warnings.warn(f"Missing keys in checkpoint: {result.missing_keys}")
    model.to(_DEVICE).bfloat16().eval()

    tokenizer_name = cfg_dict.get("tokenizer_name", config.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=_DEVICE)

    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        next_token = logits[0, -1].argmax().item()
        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token]], device=_DEVICE, dtype=torch.long),
        ], dim=1)

    return tokenizer.decode(input_ids[0].tolist())
