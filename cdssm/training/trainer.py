import math
import time
import torch
import torch.nn.functional as F

_DEVICE = torch.device("cuda")


def train_one_epoch(model, dataloader, optimizer, scheduler, grad_accum_steps=8, vocab_size=50257, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_batches = 0
    step = 0

    optimizer.zero_grad()
    start_time = time.time()
    accum_since_step = 0

    ppl_clamp = math.log(vocab_size)
    log_interval = max(10, len(dataloader) // (grad_accum_steps * 10))

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(_DEVICE, non_blocking=True)
        y = y.to(_DEVICE, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

        (loss / grad_accum_steps).backward()
        accum_since_step += 1

        total_loss += loss.item()
        total_batches += 1

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log_interval == 0:
                avg_loss = total_loss / total_batches
                ppl = math.exp(min(avg_loss, ppl_clamp))
                elapsed = time.time() - start_time
                tok_per_sec = total_batches * x.shape[0] * x.shape[1] / elapsed
                print(f"  Step {step:5d} | Task Loss {avg_loss:.4f} | PPL {ppl:.2f} | {tok_per_sec:.0f} tok/s")

    if accum_since_step % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_batches
