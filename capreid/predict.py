"""Encode a single image (or directory) with CapReID image tower."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from .models import build_capreid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    ap.add_argument("--image", required=True, help="Image path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/logs/predict.json")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_classes = int(ckpt.get("num_classes") or ckpt.get("cfg", {}).get("_num_classes", 751))
    # Market-1501 train IDs default 751
    if "num_classes" not in ckpt and "model" in ckpt:
        w = ckpt["model"].get("prompts")
        if w is not None:
            num_classes = int(w.shape[0])
    model = build_capreid(cfg, num_classes=num_classes)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    h = int(cfg["data"]["height"])
    w = int(cfg["data"]["width"])
    tfm = transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(cfg["data"]["mean"], cfg["data"]["std"]),
        ]
    )
    img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img).cpu().squeeze(0).tolist()
    payload = {"image": args.image, "dim": len(feat), "embedding": feat[:16] + ["..."]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"image": args.image, "dim": len(feat)}, indent=2))
    print(json.dumps({"image": args.image, "dim": len(feat)}, indent=2))


if __name__ == "__main__":
    main()
