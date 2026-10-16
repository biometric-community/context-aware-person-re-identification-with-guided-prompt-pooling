"""Evaluate CapReID checkpoint on Market-1501."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .data import build_dataloader
from .metrics import evaluate_rank
from .models import build_capreid
from .train import extract_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    train_loader = build_dataloader(cfg, "train")
    num_classes = train_loader.dataset.num_classes
    model = build_capreid(cfg, num_classes=num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    q = extract_features(model, build_dataloader(cfg, "query", shuffle=False), device)
    g = extract_features(model, build_dataloader(cfg, "gallery", shuffle=False), device)
    metrics = evaluate_rank(
        q["feats"],
        q["pids"],
        q["cams"],
        g["feats"],
        g["pids"],
        g["cams"],
        ranks=list(cfg["eval"].get("ranks", [1, 5, 10])),
    )
    out = Path(cfg["output"]["logs"]) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "checkpoint": args.checkpoint, "dataset": "Market-1501"}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
