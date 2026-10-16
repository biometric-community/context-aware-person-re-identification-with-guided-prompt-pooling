"""Sinkhorn Optimal Transport helpers for CapReID prompt pooling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sinkhorn(
    cost: torch.Tensor,
    eps: float = 0.05,
    max_iter: int = 50,
) -> torch.Tensor:
    """Entropic OT plan via Sinkhorn-Knopp.

    Args:
        cost: [B, N, M] pairwise costs (lower = preferred).
        eps: entropic regularization.
        max_iter: Sinkhorn iterations.

    Returns:
        pi: [B, N, M] transport plan (rows and cols approx. uniform).
    """
    # K = exp(-C / eps)
    log_k = -cost / max(eps, 1e-8)
    u = torch.zeros(cost.shape[0], cost.shape[1], device=cost.device, dtype=cost.dtype)
    v = torch.zeros(cost.shape[0], cost.shape[2], device=cost.device, dtype=cost.dtype)
    for _ in range(max_iter):
        # u = log a - logsumexp_j (log_K_ij + v_j)
        u = -torch.logsumexp(log_k + v.unsqueeze(1), dim=2)
        v = -torch.logsumexp(log_k + u.unsqueeze(2), dim=1)
    log_pi = log_k + u.unsqueeze(2) + v.unsqueeze(1)
    return log_pi.exp()


def wasserstein_cost(feat: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
    """Cosine distance cost between image feature tokens and prompts.

    Args:
        feat: [B, D] or [B, T, D] image features.
        prompts: [B, K, D] or [K, D] prompt embeddings.

    Returns:
        cost: [B, N, K] where N=1 if feat is 2D.
    """
    if feat.dim() == 2:
        feat = feat.unsqueeze(1)  # [B, 1, D]
    if prompts.dim() == 2:
        prompts = prompts.unsqueeze(0).expand(feat.size(0), -1, -1)
    f = F.normalize(feat, dim=-1)
    p = F.normalize(prompts, dim=-1)
    # cosine similarity -> cost = 1 - sim
    sim = torch.einsum("bnd,bkd->bnk", f, p)
    return 1.0 - sim


def ot_prompt_weights(
    feat: torch.Tensor,
    prompts: torch.Tensor,
    eps: float = 0.05,
    max_iter: int = 50,
) -> torch.Tensor:
    """Soft OT weights γ over K prompts for each sample.

    Returns:
        gamma: [B, K] normalized weights (sum_k gamma_ik = 1).
    """
    cost = wasserstein_cost(feat, prompts)  # [B, N, K]
    pi = sinkhorn(cost, eps=eps, max_iter=max_iter)  # [B, N, K]
    # collapse token dim N
    mass = pi.sum(dim=1)  # [B, K]
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
