"""RULER Benchmark: Long-Context Gold Standard.

4 core tasks:
1. S-NIAH: Single Needle (UUID keys)
2. MK-NIAH: Multi-Key (4 keys + 3 distractors)
3. MV-NIAH: Multi-Value (1 key → 4 values)
4. VT: Variable Tracking (chained assignments)

Reference: Hsieh et al. "RULER" (2024)
"""

import gc
import random as _random
from functools import partial

import torch

from kssm.config import KSSMConfig
from kssm.model.language_model import KSSMLMHeadModel
from experiments.metrics import compute_random_baseline
from experiments.seed import seed_everything
from experiments.training import (
    calibrate_for_synthetic,
    train_synthetic_task,
)


# Token vocabulary
PAD, KEY_START, KEY_END, VALUE_START, VALUE_END, QUERY_START, ASSIGN, CHAIN, SEP = range(9)
NOISE_START, VALUE_OFFSET = 10, 100
HEX_TO_TOKEN = {c: VALUE_OFFSET + i for i, c in enumerate("0123456789abcdef")}
_HEX_CHARS = "0123456789abcdef"


def _deterministic_hex(rng: _random.Random, length: int) -> str:
    """Generate deterministic hex string using a seeded RNG."""
    return ''.join(rng.choice(_HEX_CHARS) for _ in range(length))


def encode_uuid(s: str) -> list[int]:
    return [HEX_TO_TOKEN[c] for c in s.replace("-", "")]


def generate_sniah_batch(batch_size: int, seq_len: int, vocab_size: int = 256, device: str = "cuda"):
    """S-NIAH: Single Needle with UUID keys."""
    rng = _random.Random(torch.randint(0, 2**31, (1,)).item())
    uuid_len = 8
    tokens = torch.randint(NOISE_START, NOISE_START + 80, (batch_size, seq_len), device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        key, value = _deterministic_hex(rng, uuid_len), _deterministic_hex(rng, uuid_len)
        key_tok, val_tok = encode_uuid(key), encode_uuid(value)

        pos = torch.randint(5, seq_len // 2, (1,)).item()
        tokens[b, pos] = KEY_START
        for i, t in enumerate(key_tok): tokens[b, pos + 1 + i] = t
        tokens[b, pos + 1 + uuid_len] = KEY_END
        tokens[b, pos + 2 + uuid_len] = VALUE_START
        for i, t in enumerate(val_tok): tokens[b, pos + 3 + uuid_len + i] = t
        tokens[b, pos + 3 + 2 * uuid_len] = VALUE_END

        q_pos = seq_len - uuid_len - 2 - uuid_len
        tokens[b, q_pos] = QUERY_START
        tokens[b, q_pos + 1] = KEY_START
        for i, t in enumerate(key_tok): tokens[b, q_pos + 2 + i] = t
        tokens[b, q_pos + 2 + uuid_len] = KEY_END

        out_pos = seq_len - uuid_len
        for i, t in enumerate(val_tok):
            tokens[b, out_pos + i] = PAD
            targets[b, out_pos + i] = t
            mask[b, out_pos + i] = 1.0

    return tokens, targets, mask


def generate_mkniah_batch(batch_size: int, seq_len: int, vocab_size: int = 256, device: str = "cuda"):
    """MK-NIAH: Multi-Key with 4 targets + 3 distractors."""
    rng = _random.Random(torch.randint(0, 2**31, (1,)).item())
    uuid_len, n_keys, n_dist = 8, 4, 3
    tokens = torch.randint(NOISE_START, NOISE_START + 80, (batch_size, seq_len), device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        keys = [_deterministic_hex(rng, uuid_len) for _ in range(n_keys)]
        values = [_deterministic_hex(rng, uuid_len) for _ in range(n_keys)]
        distractors = [(_deterministic_hex(rng, uuid_len), _deterministic_hex(rng, uuid_len)) for _ in range(n_dist)]

        all_pairs = list(zip(keys, values)) + distractors
        seg_len = (seq_len - n_keys * uuid_len * 2 - 50) // len(all_pairs)

        for i, (k, v) in enumerate(all_pairs):
            pos = 5 + i * seg_len + torch.randint(0, max(1, seg_len - 25), (1,)).item()
            tokens[b, pos] = KEY_START
            for j, t in enumerate(encode_uuid(k)): tokens[b, pos + 1 + j] = t
            tokens[b, pos + 1 + uuid_len] = KEY_END
            tokens[b, pos + 2 + uuid_len] = VALUE_START
            for j, t in enumerate(encode_uuid(v)): tokens[b, pos + 3 + uuid_len + j] = t
            tokens[b, pos + 3 + 2 * uuid_len] = VALUE_END

        q_start = seq_len - n_keys * (uuid_len + 2) - n_keys * uuid_len
        pos = q_start
        tokens[b, pos] = QUERY_START
        pos += 1
        for k in keys:
            tokens[b, pos] = KEY_START
            for j, t in enumerate(encode_uuid(k)): tokens[b, pos + 1 + j] = t
            tokens[b, pos + 1 + uuid_len] = KEY_END
            pos += uuid_len + 2

        out_pos = seq_len - n_keys * uuid_len
        for v in values:
            for t in encode_uuid(v):
                tokens[b, out_pos] = PAD
                targets[b, out_pos] = t
                mask[b, out_pos] = 1.0
                out_pos += 1

    return tokens, targets, mask


def generate_mvniah_batch(batch_size: int, seq_len: int, vocab_size: int = 256, device: str = "cuda"):
    """MV-NIAH: Multi-Value (1 key → 4 values)."""
    rng = _random.Random(torch.randint(0, 2**31, (1,)).item())
    uuid_len, n_values = 8, 4
    tokens = torch.randint(NOISE_START, NOISE_START + 80, (batch_size, seq_len), device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        key = _deterministic_hex(rng, uuid_len)
        values = [_deterministic_hex(rng, uuid_len) for _ in range(n_values)]

        pos = torch.randint(5, seq_len // 2, (1,)).item()
        tokens[b, pos] = KEY_START
        for i, t in enumerate(encode_uuid(key)): tokens[b, pos + 1 + i] = t
        tokens[b, pos + 1 + uuid_len] = KEY_END
        tokens[b, pos + 2 + uuid_len] = VALUE_START
        p = pos + 3 + uuid_len
        for vi, v in enumerate(values):
            for t in encode_uuid(v): tokens[b, p] = t; p += 1
            if vi < n_values - 1: tokens[b, p] = SEP; p += 1
        tokens[b, p] = VALUE_END

        q_pos = seq_len - n_values * uuid_len - uuid_len - 4
        tokens[b, q_pos] = QUERY_START
        tokens[b, q_pos + 1] = KEY_START
        for i, t in enumerate(encode_uuid(key)): tokens[b, q_pos + 2 + i] = t
        tokens[b, q_pos + 2 + uuid_len] = KEY_END

        out_pos = seq_len - n_values * uuid_len
        for v in values:
            for t in encode_uuid(v):
                tokens[b, out_pos] = PAD
                targets[b, out_pos] = t
                mask[b, out_pos] = 1.0
                out_pos += 1

    return tokens, targets, mask


def generate_vt_batch(batch_size: int, seq_len: int, vocab_size: int = 256, device: str = "cuda"):
    """VT: Variable Tracking (chained assignments X1=A, X2=X1, X3=X2, X4=X3)."""
    rng = _random.Random(torch.randint(0, 2**31, (1,)).item())
    uuid_len, chain_len = 8, 4
    tokens = torch.randint(NOISE_START, NOISE_START + 80, (batch_size, seq_len), device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, seq_len, device=device)

    for b in range(batch_size):
        var_names = [_deterministic_hex(rng, uuid_len) for _ in range(chain_len)]
        initial_value = _deterministic_hex(rng, uuid_len)
        var_toks = [encode_uuid(v) for v in var_names]
        val_toks = encode_uuid(initial_value)

        seg_len = (seq_len - uuid_len - 20) // chain_len
        for i in range(chain_len):
            pos = 5 + i * seg_len + torch.randint(0, max(1, seg_len - 30), (1,)).item()
            tokens[b, pos] = KEY_START
            for j, t in enumerate(var_toks[i]): tokens[b, pos + 1 + j] = t
            tokens[b, pos + 1 + uuid_len] = KEY_END
            tokens[b, pos + 2 + uuid_len] = ASSIGN

            if i == 0:
                tokens[b, pos + 3 + uuid_len] = VALUE_START
                for j, t in enumerate(val_toks): tokens[b, pos + 4 + uuid_len + j] = t
                tokens[b, pos + 4 + 2 * uuid_len] = VALUE_END
            else:
                tokens[b, pos + 3 + uuid_len] = CHAIN
                tokens[b, pos + 4 + uuid_len] = KEY_START
                for j, t in enumerate(var_toks[i - 1]): tokens[b, pos + 5 + uuid_len + j] = t
                tokens[b, pos + 5 + 2 * uuid_len] = KEY_END

        q_pos = seq_len - uuid_len - uuid_len - 4
        tokens[b, q_pos] = QUERY_START
        tokens[b, q_pos + 1] = KEY_START
        for i, t in enumerate(var_toks[-1]): tokens[b, q_pos + 2 + i] = t
        tokens[b, q_pos + 2 + uuid_len] = KEY_END

        out_pos = seq_len - uuid_len
        for t in val_toks:
            tokens[b, out_pos] = PAD
            targets[b, out_pos] = t
            mask[b, out_pos] = 1.0
            out_pos += 1

    return tokens, targets, mask


# Task definitions - all use 'needle' baseline (1/16 for hex characters)
TASKS = {
    "S-NIAH": generate_sniah_batch,
    "MK-NIAH": generate_mkniah_batch,
    "MV-NIAH": generate_mvniah_batch,
    "VT": generate_vt_batch,
}


def run_ruler_benchmark(
    n_steps: int = 2000,
    seq_lengths: list[int] | None = None,
):
    """Run RULER benchmark with fixed configuration.

    Uses gradient accumulation to maintain fixed effective batch size
    regardless of sequence length, preventing batch size confounding.

    Reports baseline-relative metrics instead of arbitrary pass/fail thresholds.
    """
    seq_lens = seq_lengths if seq_lengths is not None else [4096, 8192, 16384]
    d_model, n_layers, vocab_size = 128, 4, 256
    target_batch_size = 32  # Fixed effective batch size

    model_name = "KSSMLMHeadModel"

    # All RULER tasks use hex character prediction (16 possibilities)
    random_baseline = compute_random_baseline('needle')

    print("=" * 60)
    print(f"RULER Benchmark | Model: {model_name}")
    print("=" * 60)
    print(f"Model: d={d_model}, L={n_layers}, seq_lens={seq_lens}")
    print(f"Effective batch size: {target_batch_size} (with gradient accumulation)")
    print(f"Random baseline: {random_baseline:.1%} (hex character prediction)")

    device = torch.device("cuda")

    results = {}
    for task_name, generator in TASKS.items():
        print(f"\n[{task_name}]")
        results[task_name] = {}

        for seq_len in seq_lens:
            gc.collect()
            torch.cuda.empty_cache()

            # Determine actual batch size that fits in memory
            if seq_len >= 16384:
                actual_batch = 8
            elif seq_len >= 8192:
                actual_batch = 16
            else:
                actual_batch = target_batch_size

            try:
                gen = partial(generator, batch_size=actual_batch, seq_len=seq_len, vocab_size=vocab_size, device=str(device))

                # Data-driven calibration
                print(f"    Calibrating spectral bounds...")
                bounds = calibrate_for_synthetic(gen, d_model)

                config = KSSMConfig(
                    d_model=d_model,
                    d_inner=d_model * 2,
                    n_layers=n_layers,
                    n_heads=32,
                ).with_calibration(**bounds)
                model = KSSMLMHeadModel(config, vocab_size).to(device).bfloat16()

                # Use gradient accumulation training
                train_result = train_synthetic_task(
                    model, gen, vocab_size, n_steps=n_steps,
                    target_batch_size=target_batch_size,
                    actual_batch_size=actual_batch,
                    task_type='needle',
                )
                acc = train_result["evaluation"].accuracy
                above_random = train_result["evaluation"].above_random

                print(f"  L={seq_len}: {acc:.1%} (above_random: {above_random:.1%})")
                results[task_name][seq_len] = {
                    'accuracy': acc,
                    'above_random': above_random,
                    'random_baseline': random_baseline,
                }
                del model

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  L={seq_len}: OOM")
                    results[task_name][seq_len] = {
                        'accuracy': 0.0,
                        'above_random': 0.0,
                        'random_baseline': random_baseline,
                        'error': 'OOM',
                    }
                else:
                    raise

    # Summary table
    print("\n" + "=" * 60)
    print(f"Summary: Above-Random Scores | Model: {model_name}")
    print("=" * 60)
    print(f"{'Task':<10} | " + " | ".join(f"L={s}" for s in seq_lens))
    print("-" * 60)
    for task_name in TASKS:
        scores = [results[task_name].get(s, {}).get('above_random', 0.0) for s in seq_lens]
        print(f"{task_name:<10} | " + " | ".join(f"{s:.1%}" for s in scores))

    return {"model": model_name, "results": results}


if __name__ == "__main__":
    seed_everything()
    run_ruler_benchmark()
