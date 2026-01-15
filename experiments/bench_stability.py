"""Stability stress test: KSSM vs Mamba.

Tests stability under increasing learning rates to demonstrate KSSM's
unconditional A-stability vs Mamba's conditional stability.

The Cayley discretization in KSSM guarantees eigenvalues stay on the unit circle,
while Mamba's ZOH discretization can become unstable with large learning rates.

Usage:
    python experiments/bench_stability.py [--n-steps 500] [--output results/]
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path for kssm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kssm.config import KSSMConfig
from kssm.modules.kssm_layer import KSSMLayer


def create_kssm_layer(d_model: int, d_state: int = 16, use_layernorm: bool = True) -> nn.Module:
    """Create a single KSSM layer for stability testing."""
    config = KSSMConfig(
        d_model=d_model,
        d_inner=d_model * 2,
        d_state=d_state,
        use_checkpointing=False,
    )

    class KSSMTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = KSSMLayer(config)
            self.use_layernorm = use_layernorm
            if use_layernorm:
                self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            out = self.layer(x)
            if isinstance(out, tuple):
                out = out[0]
            if self.use_layernorm:
                out = self.norm(out)
            return out

    return KSSMTestModel()


def create_mamba_layer(d_model: int, d_state: int = 16, use_layernorm: bool = True) -> nn.Module:
    """Create a single Mamba layer for stability testing."""
    from mamba_ssm import Mamba

    class MambaTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
            )
            self.use_layernorm = use_layernorm
            if use_layernorm:
                self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            out = self.layer(x)
            if self.use_layernorm:
                out = self.norm(out)
            return out

    return MambaTestModel()


def generate_copy_task_batch(batch_size: int, seq_len: int, d_model: int, device: str = "cuda"):
    """Generate a batch for the copy task.

    The copy task: Model must output the input sequence.
    This tests whether the model can maintain information over time.
    """
    # Random input tokens (one-hot style with some noise)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)
    # Target is the input itself
    target = x.clone()
    return x, target


def generate_induction_batch(batch_size: int, seq_len: int, d_model: int, device: str = "cuda"):
    """Generate a batch for a simplified induction task.

    Pattern: [A, B, ..., A] -> predict B
    The model needs to remember what followed A previously.
    """
    # Create pattern tokens
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.bfloat16)

    # For simplicity, we'll use MSE on the output vs a shifted version
    # This tests the model's ability to predict based on context
    target = torch.roll(x, shifts=-1, dims=1)
    target[:, -1, :] = 0  # Last position has no next token

    return x, target


def stability_test(
    model_type: str,
    lr: float,
    n_steps: int = 500,
    batch_size: int = 8,
    seq_len: int = 256,
    d_model: int = 128,
    d_state: int = 16,
    use_layernorm: bool = True,
    task: str = "copy",
    divergence_threshold: float = 100.0,
) -> dict:
    """Train model and check for divergence.

    Args:
        model_type: "kssm" or "mamba"
        lr: Learning rate
        n_steps: Number of training steps
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        d_state: State dimension
        use_layernorm: Whether to use LayerNorm
        task: "copy" or "induction"
        divergence_threshold: Loss threshold for divergence detection

    Returns:
        Dictionary with stability results.
    """
    gc.collect()
    torch.cuda.empty_cache()

    # Create model
    if model_type == "kssm":
        model = create_kssm_layer(d_model, d_state, use_layernorm)
    else:
        model = create_mamba_layer(d_model, d_state, use_layernorm)

    model = model.cuda().bfloat16()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Select task
    generate_batch = generate_copy_task_batch if task == "copy" else generate_induction_batch

    losses = []
    diverged = False
    diverge_step = None

    for step in range(n_steps):
        optimizer.zero_grad()

        x, target = generate_batch(batch_size, seq_len, d_model)

        try:
            output = model(x)
            loss = F.mse_loss(output, target)

            # Check for divergence
            loss_val = loss.item()

            if torch.isnan(loss) or torch.isinf(loss) or loss_val > divergence_threshold:
                diverged = True
                diverge_step = step
                break

            loss.backward()

            # Check for gradient explosion
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            if total_grad_norm > 1e6 or torch.isnan(torch.tensor(total_grad_norm)):
                diverged = True
                diverge_step = step
                break

            optimizer.step()
            losses.append(loss_val)

        except RuntimeError as e:
            if "nan" in str(e).lower() or "overflow" in str(e).lower():
                diverged = True
                diverge_step = step
                break
            raise

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "model": model_type,
        "lr": lr,
        "use_layernorm": use_layernorm,
        "diverged": diverged,
        "diverge_step": diverge_step,
        "final_loss": losses[-1] if losses else float("inf"),
        "min_loss": min(losses) if losses else float("inf"),
        "max_loss": max(losses) if losses else float("inf"),
        "n_steps_completed": len(losses),
    }

    return result


def run_stability_sweep(
    learning_rates: list = None,
    n_steps: int = 500,
    batch_size: int = 8,
    seq_len: int = 256,
    d_model: int = 128,
    d_state: int = 16,
    task: str = "copy",
    output_dir: Path = None,
):
    """Run stability sweep across learning rates.

    Tests both KSSM and Mamba at increasing learning rates.
    If both are stable with LayerNorm, retests without LayerNorm (Plan B).
    """
    if learning_rates is None:
        learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    print("=" * 80)
    print("KSSM vs Mamba: Stability Stress Test")
    print("=" * 80)
    print(f"Config: n_steps={n_steps}, batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"Task: {task}")
    print()

    # Phase 1: Test with LayerNorm
    print("Phase 1: Testing with LayerNorm")
    print("-" * 80)

    header = f"{'LR':>12} | {'KSSM':>20} | {'Mamba':>20}"
    print(header)
    print("-" * len(header))

    results_ln = []
    kssm_all_stable = True
    mamba_all_stable = True

    for lr in learning_rates:
        # Test KSSM
        kssm_result = stability_test(
            "kssm", lr, n_steps, batch_size, seq_len, d_model, d_state,
            use_layernorm=True, task=task
        )

        # Test Mamba
        mamba_result = stability_test(
            "mamba", lr, n_steps, batch_size, seq_len, d_model, d_state,
            use_layernorm=True, task=task
        )

        # Format results
        if kssm_result["diverged"]:
            kssm_str = f"DIVERGED @ {kssm_result['diverge_step']}"
            kssm_all_stable = False
        else:
            kssm_str = f"Stable (L={kssm_result['final_loss']:.4f})"

        if mamba_result["diverged"]:
            mamba_str = f"DIVERGED @ {mamba_result['diverge_step']}"
            mamba_all_stable = False
        else:
            mamba_str = f"Stable (L={mamba_result['final_loss']:.4f})"

        print(f"{lr:>12.0e} | {kssm_str:>20} | {mamba_str:>20}")

        results_ln.append({
            "lr": lr,
            "phase": "with_layernorm",
            "kssm_diverged": kssm_result["diverged"],
            "kssm_diverge_step": kssm_result["diverge_step"],
            "kssm_final_loss": kssm_result["final_loss"],
            "mamba_diverged": mamba_result["diverged"],
            "mamba_diverge_step": mamba_result["diverge_step"],
            "mamba_final_loss": mamba_result["final_loss"],
        })

    print()

    # Check if we need Plan B
    results_raw = []
    if kssm_all_stable and mamba_all_stable:
        print("Both models stable with LayerNorm. Executing Plan B: Testing raw layers...")
        print()
        print("Phase 2: Testing WITHOUT LayerNorm (Raw Layers)")
        print("-" * 80)
        print(header)
        print("-" * len(header))

        for lr in learning_rates:
            # Test KSSM without LayerNorm
            kssm_result = stability_test(
                "kssm", lr, n_steps, batch_size, seq_len, d_model, d_state,
                use_layernorm=False, task=task
            )

            # Test Mamba without LayerNorm
            mamba_result = stability_test(
                "mamba", lr, n_steps, batch_size, seq_len, d_model, d_state,
                use_layernorm=False, task=task
            )

            # Format results
            if kssm_result["diverged"]:
                kssm_str = f"DIVERGED @ {kssm_result['diverge_step']}"
            else:
                kssm_str = f"Stable (L={kssm_result['final_loss']:.4f})"

            if mamba_result["diverged"]:
                mamba_str = f"DIVERGED @ {mamba_result['diverge_step']}"
            else:
                mamba_str = f"Stable (L={mamba_result['final_loss']:.4f})"

            print(f"{lr:>12.0e} | {kssm_str:>20} | {mamba_str:>20}")

            results_raw.append({
                "lr": lr,
                "phase": "without_layernorm",
                "kssm_diverged": kssm_result["diverged"],
                "kssm_diverge_step": kssm_result["diverge_step"],
                "kssm_final_loss": kssm_result["final_loss"],
                "mamba_diverged": mamba_result["diverged"],
                "mamba_diverge_step": mamba_result["diverge_step"],
                "mamba_final_loss": mamba_result["final_loss"],
            })

        print()

    # Combine results
    all_results = results_ln + results_raw

    print("=" * 80)

    # Summary
    print("\nSummary:")

    # Find first divergence LR for each model
    kssm_diverge_lr_ln = None
    mamba_diverge_lr_ln = None
    for r in results_ln:
        if r["kssm_diverged"] and kssm_diverge_lr_ln is None:
            kssm_diverge_lr_ln = r["lr"]
        if r["mamba_diverged"] and mamba_diverge_lr_ln is None:
            mamba_diverge_lr_ln = r["lr"]

    kssm_diverge_lr_raw = None
    mamba_diverge_lr_raw = None
    for r in results_raw:
        if r["kssm_diverged"] and kssm_diverge_lr_raw is None:
            kssm_diverge_lr_raw = r["lr"]
        if r["mamba_diverged"] and mamba_diverge_lr_raw is None:
            mamba_diverge_lr_raw = r["lr"]

    print("\nWith LayerNorm:")
    if kssm_diverge_lr_ln:
        print(f"  KSSM diverges at LR = {kssm_diverge_lr_ln:.0e}")
    else:
        print(f"  KSSM: Stable across all tested LRs")

    if mamba_diverge_lr_ln:
        print(f"  Mamba diverges at LR = {mamba_diverge_lr_ln:.0e}")
    else:
        print(f"  Mamba: Stable across all tested LRs")

    if results_raw:
        print("\nWithout LayerNorm (Raw Layers):")
        if kssm_diverge_lr_raw:
            print(f"  KSSM diverges at LR = {kssm_diverge_lr_raw:.0e}")
        else:
            print(f"  KSSM: Stable across all tested LRs")

        if mamba_diverge_lr_raw:
            print(f"  Mamba diverges at LR = {mamba_diverge_lr_raw:.0e}")
        else:
            print(f"  Mamba: Stable across all tested LRs")

    # Stability advantage calculation
    if results_raw and mamba_diverge_lr_raw and not kssm_diverge_lr_raw:
        print(f"\n*** KSSM shows stability advantage: Mamba diverges at LR={mamba_diverge_lr_raw:.0e} while KSSM remains stable ***")
    elif mamba_diverge_lr_ln and not kssm_diverge_lr_ln:
        print(f"\n*** KSSM shows stability advantage: Mamba diverges at LR={mamba_diverge_lr_ln:.0e} while KSSM remains stable ***")
    elif mamba_diverge_lr_ln and kssm_diverge_lr_ln and mamba_diverge_lr_ln < kssm_diverge_lr_ln:
        print(f"\n*** KSSM shows stability advantage: Mamba diverges at LR={mamba_diverge_lr_ln:.0e} vs KSSM at LR={kssm_diverge_lr_ln:.0e} ***")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "stability_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print(f"\nResults saved to: {csv_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="KSSM vs Mamba stability stress test")
    parser.add_argument("--n-steps", type=int, default=500, help="Training steps per test")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--d-state", type=int, default=16, help="State dimension")
    parser.add_argument("--task", type=str, default="copy", choices=["copy", "induction"],
                        help="Task type")
    parser.add_argument("--output", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=None,
                        help="Learning rates to test")

    args = parser.parse_args()

    run_stability_sweep(
        learning_rates=args.learning_rates,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        d_state=args.d_state,
        task=args.task,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
