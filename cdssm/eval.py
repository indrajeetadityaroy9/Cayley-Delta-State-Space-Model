"""Module entrypoint for checkpoint evaluation."""

import argparse
import math

from torch.utils.data import DataLoader

# H100 80 GB / 26 vCPU defaults
_BATCH_SIZE = 8
_NUM_WORKERS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    from cdssm.data.datasets import build_dataset
    from cdssm.evaluation.evaluator import evaluate_epoch
    from cdssm.inference.predict import load_model_from_checkpoint

    model, tokenizer = load_model_from_checkpoint(args.checkpoint)

    data_cfg = {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-103-raw-v1",
        "context_length": model.config.context_length,
    }
    val_dataset = build_dataset(data_cfg, "validation", tokenizer_name=model.config.tokenizer_name)
    val_loader = DataLoader(
        val_dataset,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
    )

    val_loss = evaluate_epoch(model, val_loader)
    ppl_clamp = math.log(model.config.vocab_size)
    val_ppl = math.exp(min(val_loss, ppl_clamp))
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation PPL:  {val_ppl:.2f}")


if __name__ == "__main__":
    main()
