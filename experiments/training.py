"""Training pipeline: synthetic task training, calibration."""

import gc
import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from kssm.config import KSSMConfig
from kssm.model.language_model import KSSMLMHeadModel
from kssm.modules.calibration import calibrate_spectral_bounds

from experiments.metrics import (
    EvaluationResult,
    compute_accuracy,
    compute_loss,
    compute_random_baseline,
)
from experiments.seed import get_data_seed


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    ssm_lr_factor: float = 0.1,
) -> list[dict]:
    """Split model parameters into SSM and non-SSM groups with different learning rates.

    SSM parameters (dynamics, timestep, gates, selection) use a lower learning rate
    for stability, following S4/S5 convention.
    """
    ssm_keywords = [
        'dynamics_proj', 'adaptive_dt', 'log_dt_scale',
        'decay_gate_logit', 'decay_gate_proj', 'selection_B', 'selection_C', 'selection_dt',
    ]
    ssm_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ssm_keywords):
            ssm_params.append(param)
        else:
            other_params.append(param)

    return [
        {'params': other_params, 'lr': base_lr},
        {'params': ssm_params, 'lr': base_lr * ssm_lr_factor},
    ]


def train_synthetic_task(
    model: nn.Module,
    data_generator: Callable,
    vocab_size: int,
    n_steps: int = 2000,
    learning_rate: float = 1e-3,
    ssm_lr_factor: float = 0.3,
    target_batch_size: int | None = None,
    actual_batch_size: int | None = None,
    task_type: str = 'mqar',
    convergence_window: int = 100,
    convergence_threshold: float = 0.001,
) -> dict:
    """Train model on synthetic task with gradient accumulation support.

    Returns:
        Dict with keys: final_loss, converged, converged_step,
        convergence_method, evaluation (EvaluationResult)
    """
    if target_batch_size is None or actual_batch_size is None:
        grad_accum_steps = 1
    else:
        grad_accum_steps = max(1, (target_batch_size + actual_batch_size - 1) // actual_batch_size)

    param_groups = build_param_groups(model, learning_rate, ssm_lr_factor)
    optimizer = torch.optim.AdamW(param_groups, fused=True)

    # Warmup + cosine decay schedule
    warmup_steps = min(100, n_steps // 10)
    scheduler = build_cosine_schedule(optimizer, warmup_steps, n_steps)
    loss_history: list[float] = []
    last_loss, last_acc = 0.0, 0.0
    converged_step = None
    convergence_method = 'max_steps'

    random_baseline = compute_random_baseline(task_type, vocab_size=vocab_size)

    model.train()
    micro_step = 0

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_correct = 0
        accum_total = 0

        for accum_idx in range(grad_accum_steps):
            micro_step += 1

            tokens, targets, mask = data_generator()
            logits = model(tokens)

            loss = compute_loss(logits, targets, mask, vocab_size)

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            accum_loss += loss.item()

            with torch.no_grad():
                correct, total = compute_accuracy(logits, targets, mask)
                accum_correct += correct
                accum_total += total

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # NaN/Inf gradient detection
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"  WARNING: NaN/Inf gradient at step {step}, skipping update", flush=True)
            optimizer.zero_grad()
            continue

        optimizer.step()
        scheduler.step()

        last_loss = accum_loss / grad_accum_steps
        last_acc = accum_correct / accum_total if accum_total > 0 else 0.0
        loss_history.append(last_loss)

        if step % 200 == 0:
            above_random = (last_acc - random_baseline) / (1.0 - random_baseline) if random_baseline < 1.0 else 0.0
            print(f"  Step {step:4d} | Loss: {last_loss:.4f} | Acc: {last_acc:.4f} | Above Random: {above_random:.1%} | GradNorm: {grad_norm:.3f}", flush=True)

        # Plateau convergence detection
        if converged_step is None and len(loss_history) >= max(convergence_window * 2, 200):
            recent = loss_history[-convergence_window:]
            previous = loss_history[-2 * convergence_window:-convergence_window]
            recent_mean = sum(recent) / len(recent)
            previous_mean = sum(previous) / len(previous)
            if previous_mean > 0:
                relative_change = abs(recent_mean - previous_mean) / previous_mean
                if relative_change < convergence_threshold:
                    print(f"  Loss plateau detected at step {step}", flush=True)
                    converged_step = len(loss_history)
                    convergence_method = 'plateau'

    # Evaluation on isolated eval split
    model.eval()
    total_correct, total_count = 0, 0
    with torch.no_grad():
        for batch_idx in range(20):
            torch.manual_seed(get_data_seed('eval') + batch_idx)
            torch.cuda.manual_seed(get_data_seed('eval') + batch_idx)

            tokens, targets, mask = data_generator()
            logits = model(tokens)
            correct, count = compute_accuracy(logits, targets, mask)
            total_correct += correct
            total_count += count

    final_accuracy = total_correct / total_count if total_count > 0 else 0.0

    evaluation = EvaluationResult(
        accuracy=final_accuracy,
        random_baseline=random_baseline,
    )

    return {
        "final_loss": last_loss,
        "converged": converged_step is not None,
        "converged_step": converged_step,
        "convergence_method": convergence_method,
        "evaluation": evaluation,
    }


def calibrate_for_synthetic(
    data_generator: Callable,
    d_model: int,
    n_batches: int = 10,
) -> dict:
    """Calibrate spectral bounds from a synthetic task data generator.

    Uses 'calibration' split seed to ensure no overlap with training or
    evaluation data.
    """

    class SyntheticDataLoader:
        def __init__(self, gen: Callable, n: int):
            self.gen = gen
            self.n = n

        def __iter__(self):
            for batch_idx in range(self.n):
                torch.manual_seed(get_data_seed('calibration') + batch_idx)
                torch.cuda.manual_seed(get_data_seed('calibration') + batch_idx)
                tokens, targets, _ = self.gen()
                yield tokens, targets

    loader = SyntheticDataLoader(data_generator, n_batches)
    return calibrate_spectral_bounds(loader, d_model)


def run_synthetic_experiment(
    experiment_name: str,
    data_generator: Callable,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_steps: int,
    task_type: str,
    batch_size: int = 32,
    extra_info: str | None = None,
    **generator_kwargs,
) -> dict:
    """Unified helper for synthetic experiments."""
    print("=" * 60)
    print(f"{experiment_name}")
    print("=" * 60)
    if extra_info:
        print(extra_info)

    random_baseline = compute_random_baseline(task_type, vocab_size=vocab_size)
    print(f"Random baseline: {random_baseline:.1%}")

    gc.collect()
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    gen = partial(
        data_generator,
        batch_size=batch_size,
        device=str(device),
        **generator_kwargs,
    )

    print("Calibrating spectral bounds from task data...")
    bounds = calibrate_for_synthetic(gen, d_model)
    print(f"Calibrated: t=[{bounds['t_min']:.2f}, {bounds['t_max']:.2f}]")

    config = KSSMConfig(
        d_model=d_model,
        d_inner=d_model * 2,
        n_layers=n_layers,
        n_heads=n_heads,
    ).with_calibration(**bounds)

    model = KSSMLMHeadModel(config, vocab_size).to(device).bfloat16()
    model = torch.compile(model, mode="reduce-overhead")

    result = train_synthetic_task(
        model, gen, vocab_size, n_steps=n_steps,
        task_type=task_type,
    )

    eval_result = result["evaluation"]
    print("\nResults:")
    print(f"  Final Accuracy: {eval_result.accuracy:.1%}")
    print(f"  Random Baseline: {eval_result.random_baseline:.1%}")
    print(f"  Above Random: {eval_result.above_random:.1%}")
    print(f"  Relative Improvement: {eval_result.relative_improvement:.1f}x over random")
    print(f"  Convergence: {result['convergence_method']} at step {result['converged_step']}")

    return {
        "accuracy": eval_result.accuracy,
        "random_baseline": eval_result.random_baseline,
        "above_random": eval_result.above_random,
        "relative_improvement": eval_result.relative_improvement,
        "convergence_method": result["convergence_method"],
        "convergence_step": result["converged_step"],
        "evaluation_result": eval_result,
    }


def build_cosine_schedule(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
