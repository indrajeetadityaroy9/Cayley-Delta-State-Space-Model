"""Dataset and DataLoader construction for CDSSM."""

import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TokenDataset(Dataset):
    """Token-concatenated dataset from any HuggingFace text dataset.

    Streams and tokenizes up to ``num_tokens`` tokens, then stores as a
    contiguous tensor for random-access training.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        split: str,
        context_length: int,
        num_tokens: int,
        text_field: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length = context_length

        print(f"Loading {dataset_name}/{dataset_config} {split} "
              f"(streaming, target {num_tokens:,} tokens)...")
        ds = load_dataset(
            dataset_name, dataset_config, split=split,
            streaming=True,
        )

        all_tokens: list[int] = []
        for example in ds:
            text = example[text_field]
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            if len(all_tokens) >= num_tokens:
                break

        all_tokens = all_tokens[:num_tokens]
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.n_sequences = (len(self.tokens) - 1) // context_length

        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Sequences: {self.n_sequences:,}")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def _worker_init_fn(seed: int):
    """Return a worker_init_fn that seeds all RNGs including torch."""
    def init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return init


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Construct a DataLoader with reproducible worker seeding and GPU pinning."""
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn(seed),
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed + 1000)
    return DataLoader(**kwargs)
