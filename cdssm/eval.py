#!/usr/bin/env python
"""Module entrypoint for checkpoint evaluation."""

import argparse

from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    from cdssm.data import build_dataset
    from cdssm.evaluation.evaluator import evaluate_epoch
    from cdssm.evaluation.metrics import perplexity_from_loss
    from cdssm.inference.predict import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(args.checkpoint, device=args.device)

    data_cfg = {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-103-raw-v1",
        "context_length": args.context_length,
        "num_workers": args.num_workers,
    }
    val_dataset = build_dataset(data_cfg, "validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loss = evaluate_epoch(model, val_loader, args.device)
    ppl = perplexity_from_loss(val_loss, model.vocab_size)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation PPL:  {ppl:.2f}")


if __name__ == "__main__":
    main()
