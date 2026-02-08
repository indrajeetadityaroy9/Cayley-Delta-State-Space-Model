import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches
