"""Generate publication-quality figures from benchmark results.

Usage:
    python experiments/plot_results.py --input results/ --output results/figures/
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_csv(filepath: Path) -> list[dict]:
    """Load CSV file into list of dicts."""
    if not filepath.exists():
        return []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_throughput_vs_transformer(results_path: Path, output_path: Path):
    """Plot throughput comparison: KSSM vs FlashAttention Transformer."""
    data = load_csv(results_path / "kssm_vs_transformer.csv")
    if not data:
        print("No transformer benchmark data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    seq_lens = []
    kssm_throughput = []
    transformer_throughput = []
    kssm_memory = []
    transformer_memory = []

    for row in data:
        seq_len = int(row["seq_len"])
        kssm_tp = float(row["kssm_throughput"])
        trans_tp = float(row["transformer_throughput"])
        kssm_mem = float(row["kssm_memory_gb"])
        trans_mem = float(row["transformer_memory_gb"])

        seq_lens.append(seq_len)
        kssm_throughput.append(kssm_tp if kssm_tp > 0 else None)
        transformer_throughput.append(trans_tp if trans_tp > 0 else None)
        kssm_memory.append(kssm_mem if kssm_mem < float("inf") else None)
        transformer_memory.append(trans_mem if trans_mem < float("inf") else None)

    # Plot throughput
    ax1.plot(seq_lens, kssm_throughput, 'o-', label='KSSM (O(L))', linewidth=2, markersize=8, color='#2ecc71')

    # Plot transformer throughput, mark OOM points
    trans_valid = [(s, t) for s, t in zip(seq_lens, transformer_throughput) if t is not None]
    trans_oom = [(s, 0) for s, t in zip(seq_lens, transformer_throughput) if t is None]

    if trans_valid:
        ax1.plot([x[0] for x in trans_valid], [x[1] for x in trans_valid], 's-',
                 label='FlashAttention (O(L²))', linewidth=2, markersize=8, color='#3498db')
    if trans_oom:
        ax1.scatter([x[0] for x in trans_oom], [100 for _ in trans_oom],
                    marker='x', s=100, color='#e74c3c', label='Transformer OOM', linewidth=3)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax1.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Plot memory
    ax2.plot(seq_lens, kssm_memory, 'o-', label='KSSM', linewidth=2, markersize=8, color='#2ecc71')

    if trans_valid:
        trans_mem_valid = [(s, m) for s, m in zip(seq_lens, transformer_memory) if m is not None]
        if trans_mem_valid:
            ax2.plot([x[0] for x in trans_mem_valid], [x[1] for x in trans_mem_valid], 's-',
                     label='FlashAttention', linewidth=2, markersize=8, color='#3498db')

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax2.set_title('Memory Usage', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    output_file = output_path / "fig1_kssm_vs_transformer.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_throughput_vs_mamba(results_path: Path, output_path: Path):
    """Plot throughput comparison: KSSM vs Mamba."""
    data = load_csv(results_path / "kssm_vs_mamba.csv")
    if not data:
        print("No Mamba benchmark data found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    seq_lens = []
    kssm_throughput = []
    mamba_throughput = []
    kssm_memory = []
    mamba_memory = []

    for row in data:
        seq_len = int(row["seq_len"])
        kssm_tp = float(row["kssm_throughput"])
        mamba_tp = float(row["mamba_throughput"])
        kssm_mem = float(row["kssm_memory_gb"])
        mamba_mem = float(row["mamba_memory_gb"])

        seq_lens.append(seq_len)
        kssm_throughput.append(kssm_tp if kssm_tp > 0 else None)
        mamba_throughput.append(mamba_tp if mamba_tp > 0 else None)
        kssm_memory.append(kssm_mem if kssm_mem < float("inf") else None)
        mamba_memory.append(mamba_mem if mamba_mem < float("inf") else None)

    # Plot throughput
    ax1.plot(seq_lens, kssm_throughput, 'o-', label='KSSM', linewidth=2, markersize=8, color='#2ecc71')
    ax1.plot(seq_lens, mamba_throughput, 's-', label='Mamba', linewidth=2, markersize=8, color='#9b59b6')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax1.set_title('KSSM vs Mamba Throughput', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Plot memory
    ax2.plot(seq_lens, kssm_memory, 'o-', label='KSSM', linewidth=2, markersize=8, color='#2ecc71')
    ax2.plot(seq_lens, mamba_memory, 's-', label='Mamba', linewidth=2, markersize=8, color='#9b59b6')

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax2.set_title('Memory Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))

    plt.tight_layout()
    output_file = output_path / "fig2_kssm_vs_mamba.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_stability_heatmap(results_path: Path, output_path: Path):
    """Plot stability heatmap: KSSM vs Mamba across learning rates."""
    data = load_csv(results_path / "stability_sweep.csv")
    if not data:
        print("No stability data found")
        return

    # Separate data by phase
    ln_data = [r for r in data if r["phase"] == "with_layernorm"]
    raw_data = [r for r in data if r["phase"] == "without_layernorm"]

    def create_heatmap(ax, rows, title):
        if not rows:
            return

        lrs = sorted(set(float(r["lr"]) for r in rows))
        models = ["KSSM", "Mamba"]

        # Create stability matrix (1 = stable, 0 = diverged)
        matrix = np.zeros((2, len(lrs)))

        for i, lr in enumerate(lrs):
            row = next((r for r in rows if float(r["lr"]) == lr), None)
            if row:
                matrix[0, i] = 0 if row["kssm_diverged"] == "True" else 1
                matrix[1, i] = 0 if row["mamba_diverged"] == "True" else 1

        # Create heatmap
        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)

        ax.set_xlabel('Learning Rate', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(lrs)):
                status = "Stable" if matrix[i, j] == 1 else "Diverged"
                color = 'white' if matrix[i, j] == 0 else 'black'
                ax.text(j, i, status, ha='center', va='center', color=color, fontsize=9)

        return im

    # Determine subplot layout based on available data
    n_plots = sum([1 if ln_data else 0, 1 if raw_data else 0])
    if n_plots == 0:
        print("No stability data to plot")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    if ln_data:
        create_heatmap(axes[plot_idx], ln_data, "With LayerNorm")
        plot_idx += 1

    if raw_data:
        im = create_heatmap(axes[plot_idx], raw_data, "Without LayerNorm (Raw Layers)")

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn), cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Diverged', 'Stable'])

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    output_file = output_path / "fig3_stability_heatmap.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_combined_summary(results_path: Path, output_path: Path):
    """Generate a combined summary figure with all benchmarks."""
    trans_data = load_csv(results_path / "kssm_vs_transformer.csv")
    mamba_data = load_csv(results_path / "kssm_vs_mamba.csv")
    stability_data = load_csv(results_path / "stability_sweep.csv")

    if not (trans_data or mamba_data or stability_data):
        print("No data for combined summary")
        return

    fig = plt.figure(figsize=(14, 10))

    # Subplot 1: Throughput vs Sequence Length (all models)
    ax1 = fig.add_subplot(2, 2, 1)

    # KSSM from transformer benchmark
    if trans_data:
        seq_lens = [int(r["seq_len"]) for r in trans_data]
        kssm_tp = [float(r["kssm_throughput"]) for r in trans_data]
        trans_tp = [float(r["transformer_throughput"]) if float(r["transformer_throughput"]) > 0 else None
                    for r in trans_data]

        ax1.plot(seq_lens, kssm_tp, 'o-', label='KSSM', linewidth=2, markersize=6, color='#2ecc71')
        trans_valid = [(s, t) for s, t in zip(seq_lens, trans_tp) if t is not None]
        if trans_valid:
            ax1.plot([x[0] for x in trans_valid], [x[1] for x in trans_valid], 's-',
                     label='FlashAttention', linewidth=2, markersize=6, color='#3498db')

    # Mamba from mamba benchmark
    if mamba_data:
        seq_lens_m = [int(r["seq_len"]) for r in mamba_data]
        mamba_tp = [float(r["mamba_throughput"]) for r in mamba_data]
        ax1.plot(seq_lens_m, mamba_tp, '^-', label='Mamba', linewidth=2, markersize=6, color='#9b59b6')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Throughput (tok/s)')
    ax1.set_title('Throughput Comparison', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Memory vs Sequence Length
    ax2 = fig.add_subplot(2, 2, 2)

    if trans_data:
        kssm_mem = [float(r["kssm_memory_gb"]) for r in trans_data]
        trans_mem = [float(r["transformer_memory_gb"]) if float(r["transformer_memory_gb"]) < float("inf") else None
                     for r in trans_data]

        ax2.plot(seq_lens, kssm_mem, 'o-', label='KSSM', linewidth=2, markersize=6, color='#2ecc71')
        trans_mem_valid = [(s, m) for s, m in zip(seq_lens, trans_mem) if m is not None]
        if trans_mem_valid:
            ax2.plot([x[0] for x in trans_mem_valid], [x[1] for x in trans_mem_valid], 's-',
                     label='FlashAttention', linewidth=2, markersize=6, color='#3498db')

    if mamba_data:
        mamba_mem = [float(r["mamba_memory_gb"]) for r in mamba_data]
        ax2.plot(seq_lens_m, mamba_mem, '^-', label='Mamba', linewidth=2, markersize=6, color='#9b59b6')

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.set_title('Memory Comparison', fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Speedup ratio
    ax3 = fig.add_subplot(2, 2, 3)

    if trans_data:
        speedup_trans = []
        seq_lens_valid = []
        for r in trans_data:
            kssm_t = float(r["kssm_throughput"])
            trans_t = float(r["transformer_throughput"])
            if trans_t > 0:
                speedup_trans.append(kssm_t / trans_t)
                seq_lens_valid.append(int(r["seq_len"]))

        if speedup_trans:
            ax3.bar([str(s) for s in seq_lens_valid], speedup_trans, color='#3498db', alpha=0.7,
                    label='KSSM / FlashAttention')

    ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Parity')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Speedup Ratio')
    ax3.set_title('KSSM Speedup vs FlashAttention', fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Stability summary
    ax4 = fig.add_subplot(2, 2, 4)

    if stability_data:
        # Get raw layer results if available, otherwise with layernorm
        raw_data = [r for r in stability_data if r["phase"] == "without_layernorm"]
        if not raw_data:
            raw_data = [r for r in stability_data if r["phase"] == "with_layernorm"]

        if raw_data:
            lrs = [float(r["lr"]) for r in raw_data]
            kssm_stable = [0 if r["kssm_diverged"] == "True" else 1 for r in raw_data]
            mamba_stable = [0 if r["mamba_diverged"] == "True" else 1 for r in raw_data]

            x = np.arange(len(lrs))
            width = 0.35

            ax4.bar(x - width/2, kssm_stable, width, label='KSSM', color='#2ecc71', alpha=0.8)
            ax4.bar(x + width/2, mamba_stable, width, label='Mamba', color='#9b59b6', alpha=0.8)

            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel('Stable (1) / Diverged (0)')
            ax4.set_title('Stability at High Learning Rates', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels([f'{lr:.0e}' for lr in lrs], rotation=45, ha='right')
            ax4.legend(loc='best', fontsize=9)
            ax4.set_ylim(0, 1.2)
            ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No stability data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Stability Summary', fontweight='bold')

    plt.tight_layout()
    output_file = output_path / "fig_combined_summary.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--input", type=str, default="experiments/results",
                        help="Input directory with CSV results")
    parser.add_argument("--output", type=str, default="experiments/results/figures",
                        help="Output directory for figures")

    args = parser.parse_args()

    results_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Publication-Quality Figures")
    print("=" * 60)
    print(f"Input: {results_path}")
    print(f"Output: {output_path}")
    print()

    # Generate individual figures
    plot_throughput_vs_transformer(results_path, output_path)
    plot_throughput_vs_mamba(results_path, output_path)
    plot_stability_heatmap(results_path, output_path)

    # Generate combined summary
    plot_combined_summary(results_path, output_path)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
