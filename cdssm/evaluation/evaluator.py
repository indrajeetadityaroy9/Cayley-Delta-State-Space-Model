import torch
import torch.nn.functional as F

_DEVICE = torch.device("cuda")


@torch.no_grad()
def evaluate_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in dataloader:
        x = x.to(_DEVICE, non_blocking=True)
        y = y.to(_DEVICE, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches
