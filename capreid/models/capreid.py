"""CapReID model: CLIP-ViT/B-16 + OT-guided prompt pooling."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from .ot import ot_prompt_weights


class CapReID(nn.Module):
    """CLIP image encoder + per-class learnable prompts with OT pooling.

    Stage 1 trains prompts (image tower frozen). Stage 2 freezes prompts and
    fine-tunes the image encoder (+ ID classifier head).
    """

    def __init__(
        self,
        num_classes: int,
        n_pool: int = 4,
        alpha: float = 0.45,
        ot_eps: float = 0.05,
        ot_iters: int = 50,
        pretrained: str = "openai",
        model_name: str = "ViT-B-16",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.n_pool = int(n_pool)
        self.alpha = float(alpha)
        self.ot_eps = float(ot_eps)
        self.ot_iters = int(ot_iters)

        clip, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = clip.visual
        self.embed_dim = int(clip.visual.output_dim)
        # learnable prompt bank: [C, n_pool, D]
        self.prompts = nn.Parameter(
            torch.randn(self.num_classes, self.n_pool, self.embed_dim) * 0.02
        )
        self.bn = nn.BatchNorm1d(self.embed_dim)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.visual(images)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return F.normalize(feat.float(), dim=-1)

    def pool_prompts(
        self,
        image_feat: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """OT-pool class prompts and fuse with global prompt.

        Returns:
            t_final: [B, D] fused prompt embedding (L2-normalized)
            gamma: [B, n_pool] OT weights
        """
        # gather class prompts [B, K, D]
        cls_prompts = self.prompts[labels]  # [B, K, D]
        gamma = ot_prompt_weights(
            image_feat, cls_prompts, eps=self.ot_eps, max_iter=self.ot_iters
        )  # [B, K]
        t_c = torch.einsum("bk,bkd->bd", gamma, cls_prompts)
        # global prompt = mean of all class-pooled prompts (dataset-level)
        # approximate with mean over current batch class pools + full bank mean
        t_g = self.prompts.mean(dim=(0, 1))  # [D]
        t_g = t_g.unsqueeze(0).expand_as(t_c)
        t_final = self.alpha * t_c + (1.0 - self.alpha) * t_g
        return F.normalize(t_final, dim=-1), gamma

    def forward_stage1(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            img_feat = self.encode_image(images)
        t_final, gamma = self.pool_prompts(img_feat, labels)
        return {"image_feat": img_feat, "text_feat": t_final, "gamma": gamma}

    def forward_stage2(
        self, images: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        img_feat = self.encode_image(images)
        feat_bn = self.bn(img_feat)
        logits = self.classifier(feat_bn)
        out: Dict[str, torch.Tensor] = {
            "image_feat": img_feat,
            "feat_bn": feat_bn,
            "logits": logits,
        }
        if labels is not None:
            with torch.no_grad():
                t_final, gamma = self.pool_prompts(img_feat.detach(), labels)
            out["text_feat"] = t_final
            out["gamma"] = gamma
        return out

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        stage: int = 2,
    ) -> Dict[str, torch.Tensor]:
        if stage == 1:
            assert labels is not None
            return self.forward_stage1(images, labels)
        return self.forward_stage2(images, labels)

    def freeze_visual(self, freeze: bool = True) -> None:
        for p in self.visual.parameters():
            p.requires_grad = not freeze

    def freeze_prompts(self, freeze: bool = True) -> None:
        self.prompts.requires_grad = not freeze


def build_capreid(cfg: dict, num_classes: int) -> CapReID:
    m = cfg.get("model", {})
    return CapReID(
        num_classes=num_classes,
        n_pool=int(m.get("n_pool", 4)),
        alpha=float(m.get("alpha", 0.45)),
        ot_eps=float(m.get("ot_eps", 0.05)),
        ot_iters=int(m.get("ot_iters", 50)),
        pretrained=str(m.get("pretrained", "openai")),
        model_name=str(m.get("backbone", "ViT-B-16")),
    )
