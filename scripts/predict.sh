#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/public/tbiom/.venv/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IMG="${1:?usage: predict.sh <image> [config] [ckpt]}"
CFG="${2:-configs/default.yaml}"
CKPT="${3:-outputs/checkpoints/best.pt}"
exec "$PYTHON" -u -m capreid.predict --config "$CFG" --checkpoint "$CKPT" --image "$IMG" --device cuda
