"""Long Range Arena (LRA) Benchmark.

Standard benchmark for evaluating long-range dependency modeling.
Reference: Tay et al. "Long Range Arena: A Benchmark for Efficient Transformers" (2020)

Tasks implemented:
1. ListOps: Hierarchical mathematical expressions (seq_len=2048, 10 classes)
2. Text (IMDB): Byte-level sentiment classification (seq_len=1024, 2 classes)

SOTA Reference (accuracy %):
  Model        | ListOps | Text
  Transformer  | 36.37   | 64.27
  S4           | 58.35   | 76.02
  Mamba        | ~59     | ~76
  LRU          | 58.9    | 89.4
"""

import gc
import random as _random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset as hf_load_dataset

from kssm.config import KSSMConfig
from kssm.model.backbone import KSSMBackbone
from kssm.modules.components import compute_variance_preserving_std
from experiments.seed import seed_everything, init_worker_rng
from experiments.training import build_param_groups, build_cosine_schedule


# ──────────────────────────────────────────────────────────────
# Classification Model
# ──────────────────────────────────────────────────────────────

class KSSMClassifier(nn.Module):
    """Classification wrapper: KSSMBackbone + mean pooling + linear head."""

    def __init__(self, config: KSSMConfig, n_classes: int, vocab_size: int):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.backbone = KSSMBackbone(config)
        self.head = nn.Linear(config.d_model, n_classes)

        stds = compute_variance_preserving_std(
            config.d_model, config.d_inner, config.n_layers
        )
        nn.init.normal_(self.embedding.weight, std=stds["embedding"])
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, L) token IDs

        Returns:
            logits: (B, n_classes)
        """
        x = self.embedding(x)
        x = self.backbone(x)
        x = x.mean(dim=1)
        return self.head(x)


# ──────────────────────────────────────────────────────────────
# ListOps: Procedural Generation
# ──────────────────────────────────────────────────────────────

_LISTOPS_TOKENS = {
    '<pad>': 0, '[': 1, ']': 2,
    'MIN': 3, 'MAX': 4, 'MED': 5, 'SM': 6,
    '0': 7, '1': 8, '2': 9, '3': 10, '4': 11,
    '5': 12, '6': 13, '7': 14, '8': 15, '9': 16,
}
_LISTOPS_VOCAB_SIZE = 17
_LISTOPS_OPS = ['MIN', 'MAX', 'MED', 'SM']


def _generate_listops_tokens(rng: _random.Random, target_len: int = 2000, max_depth: int = 10):
    """Generate a ListOps expression as a list of string tokens."""

    def generate(depth, budget):
        if budget <= 1 or depth >= max_depth:
            return [str(rng.randint(0, 9))]

        op = rng.choice(_LISTOPS_OPS)
        n_args = rng.randint(2, min(5, max(2, budget // 3)))

        tokens = ['[', op]
        remaining = budget - 3  # account for [, op, ]

        for i in range(n_args):
            if remaining <= 0:
                break
            if i < n_args - 1:
                arg_budget = max(1, remaining // max(1, n_args - i))
                arg_budget = max(1, rng.randint(1, max(1, arg_budget * 2)))
                arg_budget = min(arg_budget, remaining)
            else:
                arg_budget = remaining

            arg = generate(depth + 1, arg_budget)
            tokens.extend(arg)
            remaining -= len(arg)

        tokens.append(']')
        return tokens

    return generate(0, target_len)


def _evaluate_listops(tokens: list[str]) -> int:
    """Evaluate a ListOps expression via stack-based parsing. Returns 0-9."""
    stack = []
    for t in tokens:
        if t == ']':
            args = []
            while stack and isinstance(stack[-1], int):
                args.append(stack.pop())
            args.reverse()
            if not stack or not isinstance(stack[-1], str):
                return 0
            op = stack.pop()
            if stack and stack[-1] == '[':
                stack.pop()
            if not args:
                stack.append(0)
                continue
            if op == 'MIN':
                stack.append(min(args))
            elif op == 'MAX':
                stack.append(max(args))
            elif op == 'MED':
                s = sorted(args)
                stack.append(s[len(s) // 2])
            elif op == 'SM':
                stack.append(sum(args) % 10)
            else:
                stack.append(0)
        elif t in _LISTOPS_OPS:
            stack.append(t)
        elif t == '[':
            stack.append('[')
        else:
            stack.append(int(t))

    return stack[0] if stack and isinstance(stack[0], int) else 0


class ListOpsDataset(Dataset):
    """Procedurally generated ListOps dataset."""

    def __init__(self, n_samples: int, seq_len: int = 2048, seed: int = 0):
        self.seq_len = seq_len
        self.tokens_list = []
        self.labels = []

        print(f"  Generating {n_samples} ListOps samples...")
        rng = _random.Random(seed)
        for i in range(n_samples):
            tokens = _generate_listops_tokens(rng, target_len=seq_len - 10)
            label = _evaluate_listops(tokens)

            indices = [_LISTOPS_TOKENS.get(t, 0) for t in tokens]
            if len(indices) < seq_len:
                indices += [0] * (seq_len - len(indices))
            else:
                indices = indices[:seq_len]

            self.tokens_list.append(torch.tensor(indices, dtype=torch.long))
            self.labels.append(label)

            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{n_samples}")

        self.tokens_tensor = torch.stack(self.tokens_list)
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)
        del self.tokens_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens_tensor[idx], self.labels_tensor[idx]


# ──────────────────────────────────────────────────────────────
# Text/IMDB: Byte-Level Sentiment Classification
# ──────────────────────────────────────────────────────────────

class IMDBByteDataset(Dataset):
    """IMDB sentiment classification with byte-level tokenization."""

    def __init__(self, split: str = 'train', seq_len: int = 1024):
        self.seq_len = seq_len
        print(f"  Loading IMDB {split} split...")
        self.dataset = hf_load_dataset('imdb', split=split)
        print(f"  Loaded {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        text_bytes = list(example['text'].encode('utf-8'))[:self.seq_len]
        if len(text_bytes) < self.seq_len:
            text_bytes += [0] * (self.seq_len - len(text_bytes))
        return (
            torch.tensor(text_bytes, dtype=torch.long),
            example['label'],
        )


# ──────────────────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────────────────

def train_lra_task(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    """Train classifier and return best test accuracy."""
    model = model.to(device).bfloat16()
    model = torch.compile(model, mode="default")

    param_groups = build_param_groups(model, base_lr=learning_rate, ssm_lr_factor=0.1)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, fused=True)
    total_steps = len(train_loader) * n_epochs
    scheduler = build_cosine_schedule(optimizer, min(500, total_steps // 10), total_steps)

    best_test_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total if total > 0 else 0.0
        train_loss = total_loss / len(train_loader)

        test_acc = evaluate_classifier(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)

        if epoch % 10 == 0 or epoch == n_epochs:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Train: {train_acc:.1%} | Test: {test_acc:.1%} | Best: {best_test_acc:.1%}")

    return best_test_acc


@torch.no_grad()
def evaluate_classifier(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate classifier accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)

    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Main Benchmark
# ──────────────────────────────────────────────────────────────

def run_lra_benchmark(
    tasks: list[str] | None = None,
    n_epochs: int = 50,
):
    """Run LRA benchmark on ListOps and/or IMDB tasks.

    Args:
        tasks: List of task names ('listops', 'imdb'). Default: both.
        n_epochs: Number of training epochs per task.
    """
    if tasks is None:
        tasks = ['listops', 'imdb']

    d_model, n_layers = 128, 4

    print("=" * 60)
    print("Long Range Arena (LRA) Benchmark | KSSM")
    print("=" * 60)
    print(f"Model: d_model={d_model}, n_layers={n_layers}")
    print(f"Tasks: {tasks}, epochs: {n_epochs}")

    device = torch.device('cuda')
    results = {}

    for task in tasks:
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n--- Task: {task} ---")

        if task == 'listops':
            train_ds = ListOpsDataset(n_samples=20000, seq_len=2048, seed=42)
            test_ds = ListOpsDataset(n_samples=2000, seq_len=2048, seed=99999)

            config = KSSMConfig(
                d_model=d_model, d_inner=d_model * 2, n_layers=n_layers, n_heads=8,
                calibrated_t_min=1.0, calibrated_t_max=2048.0,
                calibrated_freq_min=0.001, calibrated_freq_max=0.5,
            )
            model = KSSMClassifier(config, n_classes=10, vocab_size=_LISTOPS_VOCAB_SIZE)
            lr = 1e-3
            batch_size = 32

        elif task == 'imdb':
            train_ds = IMDBByteDataset('train', seq_len=1024)
            test_ds = IMDBByteDataset('test', seq_len=1024)

            config = KSSMConfig(
                d_model=d_model, d_inner=d_model * 2, n_layers=n_layers, n_heads=8,
                calibrated_t_min=1.0, calibrated_t_max=1024.0,
                calibrated_freq_min=0.001, calibrated_freq_max=0.5,
            )
            model = KSSMClassifier(config, n_classes=2, vocab_size=256)
            lr = 1e-3
            batch_size = 32

        else:
            print(f"  Unknown task: {task}, skipping")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        print(f"  Random baseline: {1.0 / (10 if task == 'listops' else 2):.1%}")

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True,
            worker_init_fn=init_worker_rng,
        )
        test_loader = DataLoader(
            test_ds, batch_size=64, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        best_acc = train_lra_task(model, train_loader, test_loader, n_epochs, lr, device)
        results[task] = best_acc
        print(f"  Best test accuracy: {best_acc:.1%}")

        del model

    # Summary with SOTA comparison
    sota_refs = {
        'listops': {'Transformer': 36.37, 'S4': 58.35, 'Mamba': 59.0, 'LRU': 58.9},
        'imdb': {'Transformer': 64.27, 'S4': 76.02, 'Mamba': 76.2, 'LRU': 89.4},
    }

    print(f"\n{'=' * 60}")
    print("LRA Results Summary")
    print(f"{'=' * 60}")
    print(f"{'Task':>10} | {'KSSM':>8} | {'Transformer':>12} | {'S4':>8} | {'Mamba':>8} | {'LRU':>8}")
    print("-" * 65)
    for task in results:
        refs = sota_refs.get(task, {})
        kssm_pct = f"{results[task] * 100:.1f}%"
        t_pct = f"{refs.get('Transformer', 0):.1f}%"
        s4_pct = f"{refs.get('S4', 0):.1f}%"
        m_pct = f"{refs.get('Mamba', 0):.1f}%"
        l_pct = f"{refs.get('LRU', 0):.1f}%"
        print(f"{task:>10} | {kssm_pct:>8} | {t_pct:>12} | {s4_pct:>8} | {m_pct:>8} | {l_pct:>8}")

    if len(results) > 1:
        avg = sum(results.values()) / len(results) * 100
        print(f"{'Average':>10} | {avg:>7.1f}%")

    return results


if __name__ == "__main__":
    seed_everything()
    run_lra_benchmark()
