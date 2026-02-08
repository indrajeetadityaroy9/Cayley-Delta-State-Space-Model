#!/usr/bin/env python
"""Deterministic reproduction entrypoint for the canonical execution path."""

import argparse
import sys

from experiments.language.wikitext import run_wikitext, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical KSSM execution path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=1024)
    args = parser.parse_args()

    seed_everything()
    run_wikitext(epochs=args.epochs, context_length=args.context_length)
    return 0


if __name__ == "__main__":
    sys.exit(main())
