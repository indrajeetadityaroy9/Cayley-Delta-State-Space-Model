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
