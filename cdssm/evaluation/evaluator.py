import torch

from cdssm.training.losses import language_model_cross_entropy

@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = language_model_cross_entropy(logits, y)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches
