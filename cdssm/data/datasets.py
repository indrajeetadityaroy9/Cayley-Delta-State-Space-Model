import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class WikiTextDataset(Dataset):
    """WikiText-103 tokenized with GPT-2 tokenizer."""

    def __init__(self, split: str, context_length: int, cache_dir: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length = context_length

        print(f"Loading wikitext-103 {split} split...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)

        print("Tokenizing...")
        texts = [ex["text"] for ex in dataset if ex["text"].strip()]
        tokenized = self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        all_tokens = [tok for ids in tokenized["input_ids"] for tok in ids]

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


class StreamingLMDataset(Dataset):
    """Token-concatenated dataset from HuggingFace streaming datasets.

    Supports FineWeb-Edu, SlimPajama, or any HuggingFace text dataset.
    Streams and tokenizes up to ``num_tokens`` tokens, then stores as a
    contiguous tensor for random-access training.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        split: str,
        context_length: int,
        num_tokens: int = 300_000_000,
        text_field: str = "text",
        cache_dir: str = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length = context_length

        print(f"Loading {dataset_name}/{dataset_config} {split} "
              f"(streaming, target {num_tokens:,} tokens)...")
        ds = load_dataset(
            dataset_name, dataset_config, split=split,
            streaming=True, cache_dir=cache_dir,
        )

        all_tokens: list[int] = []
        for example in ds:
            text = example.get(text_field, "")
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


def build_dataset(config: dict, split: str) -> Dataset:
    """Factory: build a dataset from the data section of a YAML config."""
    name = config.get("dataset_name", "wikitext")
    ctx = config["context_length"]
    cache_dir = config.get("cache_dir", None)

    if name == "wikitext":
        return WikiTextDataset(split, ctx, cache_dir=cache_dir)

    return StreamingLMDataset(
        dataset_name=name,
        dataset_config=config.get("dataset_config", "default"),
        split=split,
        context_length=ctx,
        num_tokens=config.get("num_tokens", 300_000_000),
        text_field=config.get("text_field", "text"),
        cache_dir=cache_dir,
    )
