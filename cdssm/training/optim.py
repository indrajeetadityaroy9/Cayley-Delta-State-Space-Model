import math
import torch
import torch.nn as nn

from cdssm.config.defaults import derived_ssm_lr_ratio

def build_param_groups(model: nn.Module, base_lr: float, n_layers: int) -> list[dict]:
    ssm_keywords = [
        "gate_proj",
        "log_dt_scale",
    ]
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ssm_keywords):
            ssm_params.append(param)
        else:
            other_params.append(param)

    # SSM LR = base_lr / sqrt(2 * n_layers), T-Fixup aligned depth scaling
    ssm_lr = base_lr * derived_ssm_lr_ratio(n_layers)

    return [
        {"params": other_params, "lr": base_lr},
        {"params": ssm_params, "lr": ssm_lr},
    ]


def build_cosine_schedule(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
