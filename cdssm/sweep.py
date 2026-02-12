#!/usr/bin/env python
"""Module entrypoint for local hyperparameter sweeps."""

import argparse
import itertools
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lrs", type=float, nargs="+", default=[6e-4, 4e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()

    for lr, seed in itertools.product(args.lrs, args.seeds):
        cmd = [
            sys.executable,
            "-m",
            "cdssm.train",
            "--config",
            args.config,
            "--seed",
            str(seed),
        ]
        print(f"[sweep] lr={lr} seed={seed} -> {' '.join(cmd)}")
        env = dict(os.environ)
        env["CDSSM_BASE_LR_OVERRIDE"] = str(lr)
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
