import math
import time
import torch

from cdssm.training.losses import language_model_cross_entropy, total_training_loss

def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps=8, vocab_size=50257):
    model.train()
    total_loss = 0.0
    total_batches = 0
    step = 0

    optimizer.zero_grad()
    start_time = time.time()
    accum_since_step = 0

    # Unwrap for attribute access if compiled
    orig_model = getattr(model, '_orig_mod', model)

    # Derived constants (no magic numbers)
    ppl_clamp = math.log(vocab_size)
    log_interval = max(10, len(dataloader) // (grad_accum_steps * 10))

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        task_loss = language_model_cross_entropy(logits, y)

        # Add Metabolic Loss (Parameter-Free Sparsity)
        metabolic_loss = orig_model.backbone.get_metabolic_loss()
        loss = total_training_loss(task_loss, metabolic_loss)

        (loss / grad_accum_steps).backward()
        accum_since_step += 1

        total_loss += task_loss.item() # Report task loss for PPL
        total_batches += 1

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Universal LLM standard (GPT/LLaMA/Mamba/Griffin)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log_interval == 0:
                avg_loss = total_loss / total_batches
                ppl = math.exp(min(avg_loss, ppl_clamp))
                elapsed = time.time() - start_time
                tok_per_sec = total_batches * x.shape[0] * x.shape[1] / elapsed
                print(f"  Step {step:5d} | Task Loss {avg_loss:.4f} | PPL {ppl:.2f} | Gate Cost {metabolic_loss.item():.4f} | {tok_per_sec:.0f} tok/s")

    if accum_since_step % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Universal LLM standard (GPT/LLaMA/Mamba/Griffin)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_batches
