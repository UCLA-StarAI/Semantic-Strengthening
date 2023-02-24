import torch
import torch.nn.functional as F

@torch.jit.script
def logsigmoid(x):
    return -F.softplus(-x)

@torch.jit.script
def log1mexp(x):
    lt = (x < 0.6931471805599453094).logical_and(x > 0)
    gt = x >= 0.6931471805599453094
    res = torch.empty_like(x)
    res[lt] = torch.log(-torch.expm1(-x[lt]))
    res[gt] = torch.log1p(-torch.exp(-x[gt]))
    res = res.masked_fill_(x == 0, -float('inf'))
    return res

@torch.jit.script
def logsubexp(x, y):
    delta = torch.where((x == y) & (x.isfinite() | (x < 0)), torch.zeros_like(x - y), (x-y).abs())
    return torch.max(x, y) + log1mexp(delta)

from torch import Tensor
@torch.jit.script
def logsumexp(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        m = m.masked_fill_(torch.isneginf(m), 0.)

    z = (tensor - m).exp_().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = z.masked_fill_(mask, 1.).log_().add_(m)
    z = z.masked_fill_(mask, -float('inf'))

    if not keepdim:
        z = z.squeeze(dim=dim)
    return z.clamp(max=-1.1920928955078125e-07)

@torch.jit.script
def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    with torch.no_grad():
        m = torch.maximum(x, y)
        mask = ~torch.isneginf(m)

    delta = x[mask] - y[mask]
    return delta.abs().neg().exp().log1p().add(m)

@torch.jit.script
def xty(x, y):
    return torch.where(x == 0, torch.zeros_like(x), x * y)
