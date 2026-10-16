"""Two-stage CapReID training on Market-1501."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.cuda import amp

from .data import build_dataloader
from .losses import TripletLoss, info_nce_bidirectional, prompt_aware_triplet
from .metrics import evaluate_rank
from .models import build_capreid


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dirs(cfg: dict) -> None:
    for k in ("checkpoints", "logs", "figures"):
        Path(cfg["output"][k]).mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def extract_features(model, loader, device: str) -> Dict[str, Any]:
    model.eval()
    feats, pids, cams = [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        out = model.forward_stage2(img)
        feats.append(out["image_feat"].cpu())
        pids.append(batch["pid"].numpy())
        cams.append(batch["camid"].numpy())
    import numpy as np

    return {
        "feats": torch.cat(feats, 0),
        "pids": np.concatenate(pids),
        "cams": np.concatenate(cams),
    }


def run_eval(model, cfg, device: str) -> Dict[str, float]:
    q_loader = build_dataloader(cfg, "query", shuffle=False)
    g_loader = build_dataloader(cfg, "gallery", shuffle=False)
    q = extract_features(model, q_loader, device)
    g = extract_features(model, g_loader, device)
    return evaluate_rank(
        q["feats"],
        q["pids"],
        q["cams"],
        g["feats"],
        g["pids"],
        g["cams"],
        ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
    )


@torch.no_grad()
def _cache_train_image_features(model, loader, device: str):
    """Encode train set once (visual frozen) — CLIP-ReID Stage-1 pattern."""
    model.eval()
    feats, labels = [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        feats.append(model.encode_image(img).cpu())
        labels.append(batch["label"])
    return torch.cat(feats, 0), torch.cat(labels, 0)


def train_stage1(model, loader, cfg, device: str) -> list:
    model.freeze_visual(True)
    model.freeze_prompts(False)
    print("[stage1] caching frozen image features…", flush=True)
    feat_cpu, label_cpu = _cache_train_image_features(model, loader, device)
    feat_all = feat_cpu.to(device)
    label_all = label_cpu.to(device)
    print(f"[stage1] cached {feat_all.size(0)} features", flush=True)

    params = [model.prompts]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["stage1_epochs"])
    tau = float(cfg["train"]["temperature"])
    batch = int(cfg["train"]["batch_size"])
    history = []
    n = feat_all.size(0)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        model.train()
        order = torch.randperm(n, device=device)
        n_steps = max(1, n // batch)
        for step in range(n_steps):
            idx = order[step * batch : (step + 1) * batch]
            img_feat = feat_all[idx]
            labels = label_all[idx]
            opt.zero_grad(set_to_none=True)
            t_final, _ = model.pool_prompts(img_feat, labels)
            loss = info_nce_bidirectional(img_feat, t_final, labels, temperature=tau)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if (step + 1) % int(cfg["train"].get("log_every", 50)) == 0:
                print(
                    f"[stage1] epoch {epoch}/{epochs} step {step+1}/{n_steps} loss={loss.item():.4f}",
                    flush=True,
                )
        mean_loss = sum(losses) / max(len(losses), 1)
        history.append({"epoch": epoch, "stage": 1, "loss": mean_loss, "sec": time.time() - t0})
        print(f"[stage1] epoch {epoch} mean_loss={mean_loss:.4f} ({history[-1]['sec']:.1f}s)", flush=True)
    return history


def train_stage2(model, loader, cfg, device: str) -> list:
    model.freeze_visual(False)
    model.freeze_prompts(True)
    model.train()
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["stage2_epochs"])
    margin = float(cfg["train"]["margin"])
    id_crit = nn.CrossEntropyLoss()
    tri = TripletLoss(margin=margin)
    history = []
    scaler = amp.GradScaler(enabled=device.startswith("cuda"))
    best_map = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        model.train()
        for step, batch in enumerate(loader, 1):
            img = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast(enabled=device.startswith("cuda")):
                out = model.forward_stage2(img, labels)
                loss_id = id_crit(out["logits"], labels)
                loss_t = tri(out["image_feat"], labels)
                loss_pt = prompt_aware_triplet(
                    out["image_feat"], out["text_feat"], labels, margin=margin
                )
                loss = loss_id + loss_t + loss_pt
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            if step % int(cfg["train"].get("log_every", 50)) == 0:
                print(
                    f"[stage2] epoch {epoch}/{epochs} step {step} "
                    f"loss={loss.item():.4f} id={loss_id.item():.3f} "
                    f"t={loss_t.item():.3f} pt={loss_pt.item():.3f}",
                    flush=True,
                )
        mean_loss = sum(losses) / max(len(losses), 1)
        row = {"epoch": epoch, "stage": 2, "loss": mean_loss, "sec": time.time() - t0}
        eval_every = int(cfg["train"].get("eval_every", 10))
        if epoch % eval_every == 0 or epoch == epochs:
            metrics = run_eval(model, cfg, device)
            row.update(metrics)
            print(f"[stage2] epoch {epoch} eval {metrics}", flush=True)
            ckpt_dir = Path(cfg["output"]["checkpoints"])
            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "cfg": cfg,
                "num_classes": int(model.num_classes),
            }
            torch.save(payload, ckpt_dir / f"stage2_epoch_{epoch}.pt")
            if metrics.get("mAP", -1) > best_map:
                best_map = metrics["mAP"]
                torch.save(payload, ckpt_dir / "best.pt")

        history.append(row)
        print(f"[stage2] epoch {epoch} mean_loss={mean_loss:.4f} ({row['sec']:.1f}s)", flush=True)
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CapReID")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--stage", type=str, default="all", choices=["all", "1", "2"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    _ensure_dirs(cfg)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable; falling back to CPU", flush=True)

    train_loader = build_dataloader(cfg, "train")
    num_classes = train_loader.dataset.num_classes
    print(f"Market train: {len(train_loader.dataset)} images, {num_classes} ids", flush=True)

    model = build_capreid(cfg, num_classes=num_classes).to(device)
    history = []
    if args.stage in ("all", "1"):
        history += train_stage1(model, train_loader, cfg, device)
        torch.save(
            {"model": model.state_dict(), "cfg": cfg, "num_classes": num_classes},
            Path(cfg["output"]["checkpoints"]) / "stage1_last.pt",
        )
    if args.stage in ("all", "2"):
        history += train_stage2(model, train_loader, cfg, device)

    summary = {
        "num_classes": num_classes,
        "train_size": len(train_loader.dataset),
        "device": device,
        "history": history,
        "config": args.config,
    }
    if history:
        last_with_map = [h for h in history if "mAP" in h]
        if last_with_map:
            summary["final_metrics"] = {
                k: last_with_map[-1][k]
                for k in last_with_map[-1]
                if k in ("mAP", "Rank-1", "Rank-5", "Rank-10")
            }
    out = Path(cfg["output"]["logs"]) / "train_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
