"""CapReID losses: InfoNCE, ID CE, triplet, prompt-aware triplet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_bidirectional(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised bidirectional InfoNCE (i2t + t2i) with same-id positives."""
    img = F.normalize(image_feat, dim=-1)
    txt = F.normalize(text_feat, dim=-1)
    logits = img @ txt.t() / temperature  # [B, B]
    # positives: same identity
    pos = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()
    # mask self for numerical stability when identical feats
    eye = torch.eye(pos.size(0), device=pos.device, dtype=pos.dtype)
    pos = pos * (1.0 - eye) + eye  # always include diagonal
    # i2t
    log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
    mean_log_prob_pos = (pos * log_prob).sum(1) / pos.sum(1).clamp_min(1.0)
    loss_i2t = -mean_log_prob_pos.mean()
    # t2i
    logits_t = logits.t()
    log_prob_t = logits_t - logits_t.logsumexp(dim=1, keepdim=True)
    mean_log_prob_pos_t = (pos * log_prob_t).sum(1) / pos.sum(1).clamp_min(1.0)
    loss_t2i = -mean_log_prob_pos_t.mean()
    return loss_i2t + loss_t2i


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=-1)
        dist = torch.cdist(feats, feats, p=2)
        labels = labels.view(-1)
        mask_pos = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        mask_neg = ~mask_pos
        eye = torch.eye(dist.size(0), device=dist.device, dtype=torch.bool)
        mask_pos = mask_pos & ~eye
        if mask_pos.sum() == 0 or mask_neg.sum() == 0:
            return dist.new_zeros(())
        # hardest positive / negative
        dist_ap = dist.clone()
        dist_ap[~mask_pos] = -1.0
        hardest_pos, _ = dist_ap.max(dim=1)
        dist_an = dist.clone()
        dist_an[~mask_neg] = 1e6
        hardest_neg, _ = dist_an.min(dim=1)
        valid = (hardest_pos > -0.5) & (hardest_neg < 1e5)
        if valid.sum() == 0:
            return dist.new_zeros(())
        return F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin).mean()


def prompt_aware_triplet(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """L_PT: image anchor, matched prompt positive, other-class prompts negatives."""
    img = F.normalize(image_feat, dim=-1)
    txt = F.normalize(text_feat, dim=-1)
    # positive distance
    d_pos = (img - txt).pow(2).sum(-1).sqrt()
    # negatives: all other batch texts with different labels
    dist = torch.cdist(img, txt, p=2)  # [B, B]
    mask_neg = ~labels.unsqueeze(1).eq(labels.unsqueeze(0))
    dist_neg = dist.clone()
    dist_neg[~mask_neg] = 1e6
    hardest_neg, _ = dist_neg.min(dim=1)
    valid = hardest_neg < 1e5
    if valid.sum() == 0:
        return img.new_zeros(())
    return F.relu(d_pos[valid] - hardest_neg[valid] + margin).mean()
