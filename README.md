# CapReID — Context-Aware Person Re-Identification With Guided Prompt Pooling

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of **CapReID** (Murali & Mishra, TBIOM 2026): CLIP-ViT/B-16 with **Optimal-Transport-guided prompt pooling** and a two-stage train schedule (prompt learning → image fine-tune with prompt-aware triplet).

**Repository:** https://github.com/biometric-community/context-aware-person-re-identification-with-guided-prompt-pooling

**Paper:** [10.1109/TBIOM.2026.3654624](https://doi.org/10.1109/TBIOM.2026.3654624)  
**Monorepo path:** `projects/papers/context-aware-person-re-identification-with-guided-prompt-pooling/`

## Setup

```bash
bash scripts/setup_env.sh
# uses monorepo shared .venv; requires open-clip-torch
```

## Data

Point configs at Market-1501 (not shipped in this repo):

```text
projects/datasets/market1501/extracted/Market-1501-v15.09.15/
```

DukeMTMC / MSMT17 / Occluded-Duke are optional and currently unavailable in our layout — see `DEVIATIONS.md`.

## Train / Eval / Report / Predict

```bash
# smoke (small subset)
bash scripts/train.sh configs/smoke.yaml

# full Market-1501 (< 5 GiB size gate → full protocol)
bash scripts/train_full.sh

bash scripts/eval.sh
bash scripts/report.sh
bash scripts/predict.sh /path/to/image.jpg
```

Metrics and figures land under `outputs/logs/` and `outputs/figures/`; narrative in `REPORT.md`.

## Layout

```text
capreid/          # package (models, OT, losses, Market loader, train/eval/report)
configs/          # default.yaml (full), smoke.yaml
scripts/          # shell entrypoints
DEVIATIONS.md     SOURCE_CODE.md  FIDELITY_AUDIT.md  IMPLEMENTATION_PLAN.md
```

## Citation

```bibtex
@article{murali2026capreid,
  title={Context-Aware Person Re-Identification With Guided Prompt Pooling},
  author={Murali, Nirmala and Mishra, Deepak},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2026},
  doi={10.1109/TBIOM.2026.3654624}
}
```
