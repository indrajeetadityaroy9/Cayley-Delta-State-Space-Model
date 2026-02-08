import math
import torch
import torch.nn as nn

def build_param_groups(model: nn.Module, base_lr: float) -> list[dict]:
    ssm_keywords = [
        "dynamics_proj",
        "adaptive_dt",
        "log_dt_scale",
        "utility_gate",
        "selection_B",
        "selection_C",
        "selection_dt",
    ]
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ssm_keywords):
            ssm_params.append(param)
        else:
            other_params.append(param)
    return [
        {"params": other_params, "lr": base_lr},
        {"params": ssm_params, "lr": base_lr * 0.1},
    ]


def build_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
