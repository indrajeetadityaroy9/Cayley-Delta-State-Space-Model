"""Checkpoint loading and greedy generation for CD-SSM."""

import torch
from transformers import AutoTokenizer

from cdssm.config.model import CDSSMConfig, COMPUTED_FIELDS
from cdssm.models.model import CDSSMLMHeadModel


def load_model_from_checkpoint(checkpoint_path: str) -> tuple[CDSSMLMHeadModel, AutoTokenizer]:
    """Load a trained CDSSM model and its tokenizer from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    cfg_dict = ckpt["config"]
    init_kwargs = {
        k: v for k, v in cfg_dict.items()
        if k in CDSSMConfig.__dataclass_fields__ and k not in COMPUTED_FIELDS
    }
    config = CDSSMConfig(**init_kwargs)
    model = CDSSMLMHeadModel(config)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    model.cuda().bfloat16().eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def generate_greedy(
    model: CDSSMLMHeadModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
) -> str:
    """Generate text greedily from a prompt.

    Respects the model's context_length window and stops at EOS.
    """
    context_length = model.config.context_length
    eos_id = tokenizer.eos_token_id

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) >= context_length:
        ids = ids[-(context_length - 1):]

    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= context_length:
            input_ids = input_ids[:, -context_length:]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        next_token = logits[0, -1].argmax().item()

        if next_token == eos_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[next_token]], device="cuda", dtype=torch.long),
        ], dim=1)

    return tokenizer.decode(input_ids[0].tolist())
