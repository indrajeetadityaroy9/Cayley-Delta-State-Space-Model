"""Learning Rate Sweep: A-Stability Demonstration.

Tests KSSM stability at extreme learning rates.

Stability is measured by:
1. Whether training completes without NaN/Inf
2. Final loss relative to initial loss (stability ratio)
3. Maximum loss during training (peak instability)

No arbitrary pass/fail thresholds - reports raw stability metrics.
"""

import gc
import math
from dataclasses import dataclass

import torch

from kssm.config import KSSMConfig
from kssm.model.language_model import KSSMLMHeadModel
from experiments.metrics import compute_loss, compute_random_baseline
from experiments.seed import get_data_seed, seed_everything


@dataclass
class StabilityResult:
    """Stability test result with meaningful metrics."""
    learning_rate: float
    completed: bool  # Did training complete without NaN/error?
    final_loss: float
    initial_loss: float
    max_loss: float
    min_loss: float
    steps_completed: int
    total_steps: int
    failure_reason: str | None = None

    # Penalty cap for failed/diverged runs (keeps heatmaps clean)
    _PENALTY_CAP = 10.0

    @property
    def stability_ratio(self) -> float:
        """Ratio of final to initial loss. <1 means improvement, >1 means degradation.

        Capped at 10.0 for NaN/Inf/failed runs to enable clean visualization.
        """
        # Handle invalid initial loss
        if self.initial_loss <= 0 or math.isnan(self.initial_loss):
            return self._PENALTY_CAP

        # Handle divergence / incomplete / infinite final
        if not self.completed or math.isinf(self.final_loss) or math.isnan(self.final_loss):
            return self._PENALTY_CAP

        # Calculate ratio and cap extreme values
        ratio = self.final_loss / self.initial_loss
        return min(ratio, self._PENALTY_CAP)

    @property
    def peak_instability(self) -> float:
        """Max loss relative to initial. High values indicate instability spikes.

        Capped at 10.0 for NaN/Inf/failed runs to enable clean visualization.
        """
        # Handle invalid initial loss
        if self.initial_loss <= 0 or math.isnan(self.initial_loss):
            return self._PENALTY_CAP

        # Handle divergence / incomplete / infinite max
        if not self.completed or math.isinf(self.max_loss) or math.isnan(self.max_loss):
            return self._PENALTY_CAP

        # Calculate ratio and cap extreme values
        ratio = self.max_loss / self.initial_loss
        return min(ratio, self._PENALTY_CAP)

    @property
    def converged(self) -> bool:
        """Did training converge to lower loss than initial?"""
        return self.completed and self.final_loss < self.initial_loss


def generate_induction_batch(batch_size, seq_len, vocab_size=64, device="cuda"):
    """Generate induction heads batch."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.zeros_like(tokens)
    mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        key_pos = torch.randint(0, max(1, seq_len // 4 - 1), (1,)).item()
        key = torch.randint(0, vocab_size, (1,)).item()
        value = torch.randint(0, vocab_size, (1,)).item()

        tokens[b, key_pos] = key
        tokens[b, key_pos + 1] = value

        repeat_pos = torch.randint(seq_len // 2, seq_len - 1, (1,)).item()
        tokens[b, repeat_pos] = key
        targets[b, repeat_pos] = value
        mask[b, repeat_pos] = 1.0

    return tokens, targets, mask


def run_stability_test(model, learning_rate, n_steps, batch_size, seq_len, vocab_size) -> StabilityResult:
    """Run stability test and return detailed metrics."""
    model = model.cuda().bfloat16()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)

    initial_loss = None
    max_loss = 0.0
    min_loss = float('inf')
    current_loss = 0.0

    for step in range(n_steps):
        optimizer.zero_grad()
        tokens, targets, mask = generate_induction_batch(batch_size, seq_len, vocab_size, device="cuda")

        try:
            logits = model(tokens)
            loss = compute_loss(logits, targets, mask, vocab_size)
            current_loss = loss.item()

            # Track initial loss
            if initial_loss is None:
                initial_loss = current_loss

            # Track min/max for stability analysis
            max_loss = max(max_loss, current_loss)
            min_loss = min(min_loss, current_loss)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                return StabilityResult(
                    learning_rate=learning_rate, completed=False,
                    final_loss=float('inf'), initial_loss=initial_loss or 0,
                    max_loss=max_loss, min_loss=min_loss,
                    steps_completed=step, total_steps=n_steps,
                    failure_reason=f"NaN/Inf at step {step}"
                )

            loss.backward()
            optimizer.step()

        except RuntimeError as e:
            return StabilityResult(
                learning_rate=learning_rate, completed=False,
                final_loss=float('inf'), initial_loss=initial_loss or 0,
                max_loss=max_loss, min_loss=min_loss,
                steps_completed=step, total_steps=n_steps,
                failure_reason=f"RuntimeError at step {step}: {str(e)}"
            )

    return StabilityResult(
        learning_rate=learning_rate, completed=True,
        final_loss=current_loss, initial_loss=initial_loss or current_loss,
        max_loss=max_loss, min_loss=min_loss,
        steps_completed=n_steps, total_steps=n_steps
    )


def _run_single_depth(
    n_layers: int,
    learning_rates: list[float],
    n_steps: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
) -> dict:
    """Run LR sweep for a single depth configuration."""
    results: list[StabilityResult] = []

    for learning_rate in learning_rates:
        gc.collect()
        torch.cuda.empty_cache()

        # NOTE: Using FIXED calibration bounds for stability testing.
        # Unlike other experiments which use data-driven calibration, we use
        # hardcoded bounds here to ensure consistent comparison across learning
        # rates. Data-driven calibration would vary with the random seed used
        # for calibration data, which would confound the stability measurement.
        config = KSSMConfig(
            d_model=d_model,
            d_inner=d_model * 2,
            n_layers=n_layers,
            n_heads=8,
            calibrated_t_min=1.0,
            calibrated_t_max=1000.0,
            calibrated_freq_min=0.01,
            calibrated_freq_max=0.5,
        )

        model = KSSMLMHeadModel(config, vocab_size)
        result = run_stability_test(model, learning_rate, n_steps, batch_size, seq_len, vocab_size)
        results.append(result)

        if result.completed:
            status = "OK"
            loss_str = f"{result.final_loss:.4f}"
            ratio_str = f"{result.stability_ratio:.2f}"
            peak_str = f"{result.peak_instability:.1f}x"
        else:
            status = "FAILED"
            loss_str = "---"
            ratio_str = "---"
            peak_str = "---"

        print(f"  L={n_layers:<3} | {learning_rate:>10.0e} | {status:>10} | {loss_str:>10} | {ratio_str:>8} | {peak_str:>8}")

        del model

    completed_count = sum(1 for r in results if r.completed)
    converged_count = sum(1 for r in results if r.converged)
    max_completed_lr = max((r.learning_rate for r in results if r.completed), default=0)
    max_converged_lr = max((r.learning_rate for r in results if r.converged), default=0)

    completed_results = [r for r in results if r.completed]
    if completed_results:
        avg_stability_ratio = sum(r.stability_ratio for r in completed_results) / len(completed_results)
        avg_peak_instability = sum(r.peak_instability for r in completed_results) / len(completed_results)
    else:
        avg_stability_ratio = float('inf')
        avg_peak_instability = float('inf')

    return {
        'n_layers': n_layers,
        'results': results,
        'completed_count': completed_count,
        'converged_count': converged_count,
        'max_completed_lr': max_completed_lr,
        'max_converged_lr': max_converged_lr,
        'avg_stability_ratio': avg_stability_ratio,
        'avg_peak_instability': avg_peak_instability,
    }


def run_lr_sweep(
    learning_rates: list[float] | None = None,
    n_steps: int = 1000,
    layer_depths: list[int] | None = None,
):
    """Run learning rate sweep for A-stability analysis across multiple depths.

    Tests the core A-stability claim: Cayley discretization should maintain
    stability even at high learning rates. Sweeping depth verifies that
    stability does not degrade as the model gets deeper.

    Reports detailed stability metrics instead of pass/fail:
    - Completion rate (training finished without errors)
    - Stability ratio (final/initial loss)
    - Peak instability (max loss spike during training)
    - Convergence rate (how many LRs actually improved loss)
    """
    if learning_rates is None:
        learning_rates = [1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 0.75, 1.0]
    if layer_depths is None:
        layer_depths = [1, 4, 8, 12]
    batch_size, seq_len = 8, 256
    d_model, vocab_size = 128, 64

    model_name = "KSSMLMHeadModel"

    random_baseline = compute_random_baseline('induction', vocab_size=vocab_size)

    print("=" * 60)
    print(f"Learning Rate Sweep: A-Stability Analysis | Model: {model_name}")
    print("=" * 60)
    print(f"n_steps: {n_steps}, seq_len: {seq_len}, d_model: {d_model}")
    print(f"Layer depths: {layer_depths}")
    print(f"Learning rates: {learning_rates}")
    print(f"Random baseline (induction task): {random_baseline:.1%}")

    print(f"\n{'Depth':>5} | {'LR':>10} | {'Status':>10} | {'Final Loss':>10} | {'Ratio':>8} | {'Peak':>8}")
    print("-" * 70)

    all_results = {}
    for n_layers in layer_depths:
        depth_result = _run_single_depth(
            n_layers, learning_rates, n_steps,
            batch_size, seq_len, d_model, vocab_size,
        )
        all_results[n_layers] = depth_result

    # Summary table: depth vs max stable LR
    print("\n" + "=" * 60)
    print(f"Multi-Depth Stability Summary: {model_name}")
    print("=" * 60)
    print(f"{'Depth':>6} | {'Completed':>10} | {'Converged':>10} | {'Max LR (ok)':>12} | {'Max LR (conv)':>14} | {'Avg Ratio':>10}")
    print("-" * 70)
    for n_layers in layer_depths:
        r = all_results[n_layers]
        print(f"{n_layers:>6} | {r['completed_count']:>5}/{len(learning_rates):<4} | {r['converged_count']:>5}/{len(learning_rates):<4} | {r['max_completed_lr']:>12.2e} | {r['max_converged_lr']:>14.2e} | {r['avg_stability_ratio']:>10.2f}")

    print(f"\nA-Stability Claim: If Cayley discretization provides unconditional")
    print(f"stability, max converged LR should not degrade significantly with depth.")

    return {
        'model': model_name,
        'results_by_depth': all_results,
    }


if __name__ == "__main__":
    seed_everything()
    run_lr_sweep()
