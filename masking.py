import torch

def mask_batch(batch, mask_prob, mask_token):
    if mask_prob <= 0:
        return batch
    masked = batch.clone()
    rand = torch.rand_like(batch, dtype=torch.float)
    mask = (rand < mask_prob) & (batch != 0)
    masked[mask] = mask_token
    print("masked ", mask.sum().item(), " tokens")
    return masked
