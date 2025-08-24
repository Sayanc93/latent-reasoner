import torch
from typing import Tuple

# ---------------------------- Sampling ---------------------------- #

def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> Tuple[int, float]:
    """Sample a single index from [V] logits; returns (idx, logp)"""
    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        k = int((csum > top_p).nonzero(as_tuple=True)[0][0].item()) + 1 if (csum > top_p).any() else probs.numel()
        sorted_probs = sorted_probs[:k]
        sorted_idx = sorted_idx[:k]
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1).item()
        idx = int(sorted_idx[choice].item())
        logp = float(torch.log(probs[idx] + 1e-12).item())
        return idx, logp
    
    # full multinomial
    idx = int(torch.multinomial(probs, 1).item())
    logp = float(torch.log(probs[idx] + 1e-12).item())
    return idx, logp