"""Config-only CLI for CDSSM."""

import argparse

from cdssm.config.experiment import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="cdssm", description="Cayley-Delta State Space Model")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd, help_text in [
        ("train", "Train a model"),
        ("eval", "Evaluate checkpoint perplexity"),
        ("bench", "Run lm-eval-harness benchmarks"),
    ]:
        sub.add_parser(cmd, help=help_text).add_argument(
            "--config", type=str, required=True,
        )

    args = parser.parse_args()
    exp = load_config(args.config)

    if args.command == "train":
        from cdssm.training.trainer import Trainer
        Trainer.from_config(exp).fit(exp.training.epochs)
    elif args.command == "eval":
        from cdssm.evaluation.evaluator import run_eval
        run_eval(exp)
    elif args.command == "bench":
        from cdssm.evaluation.evaluator import run_bench
        run_bench(exp)
