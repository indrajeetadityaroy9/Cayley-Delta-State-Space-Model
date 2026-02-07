#!/usr/bin/env python
"""KSSM Experiment Reproduction Script."""

import argparse
import shutil
import sys
import traceback
from pathlib import Path

from experiments.seed import seed_everything
from experiments.synthetic.mqar import run_mqar
from experiments.synthetic.selective_copying import run_selective_copying
from experiments.synthetic.ruler_benchmark import run_ruler_benchmark
from experiments.systems.scaling_benchmark import run_scaling_benchmark
from experiments.language.wikitext import run_wikitext
from experiments.stability.lr_sweep import run_lr_sweep
from experiments.synthetic.lra_benchmark import run_lra_benchmark

EXPERIMENTS = {
    "mqar": run_mqar,
    "selective_copying": run_selective_copying,
    "ruler": run_ruler_benchmark,
    "lra": run_lra_benchmark,
    "scaling_benchmark": run_scaling_benchmark,
    "wikitext": run_wikitext,
    "lr_sweep": run_lr_sweep,
}


def main():
    parser = argparse.ArgumentParser(description="KSSM Experiment Reproduction Script")
    parser.add_argument("-e", "--experiment", type=str, help="Run specific experiment")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--clean", action="store_true", help="Delete artifacts before running")
    args = parser.parse_args()

    if args.clean:
        base = Path(__file__).parent
        for d in ["checkpoints", "results", "experiments/results", "logs", "build"]:
            p = base / d
            if p.exists():
                shutil.rmtree(p)
                print(f"Deleted: {p}")
        for egg_info in base.glob("*.egg-info"):
            shutil.rmtree(egg_info)
        for pycache in base.rglob("__pycache__"):
            shutil.rmtree(pycache)
        print("Artifact cleanup complete.")
        if not (args.experiment or args.all):
            return 0

    if args.all and not args.clean:
        print("ERROR: --all requires --clean for reproducibility.")
        return 1

    names = [args.experiment] if args.experiment else list(EXPERIMENTS) if args.all else []
    if not names:
        parser.print_help()
        return 0

    results = {}
    for name in names:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS)}")
            return 1
        seed_everything()
        EXPERIMENTS[name]()
        results[name] = True
        print(f"\n[SUCCESS] {name}")

    if len(results) > 1:
        print(f"\nSummary: {sum(results.values())}/{len(results)} passed")
        for name, ok in results.items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
