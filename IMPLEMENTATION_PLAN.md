# Implementation Plan — CapReID

## Paper
- **Title:** Context-Aware Person Re-Identification With Guided Prompt Pooling
- **DOI:** 10.1109/TBIOM.2026.3654624
- **Slug:** `context-aware-person-re-identification-with-guided-prompt-pooling`

## Method summary
1. CLIP-ViT/B-16 image encoder + per-identity learnable prompt bank (`n_pool=4`).
2. **Stage 1:** freeze visual; OT/Sinkhorn-weighted prompt pooling (ε=0.05, 50 iters) fusing class + global prompts; bidirectional InfoNCE (τ=0.07).
3. **Stage 2:** freeze prompts; fine-tune image encoder with ID CE + triplet + prompt-aware triplet (margin 0.3).

## Figure inventory (Results → our plots)
| Paper fig | Type | Our artifact |
|-----------|------|--------------|
| Fig. 1 | architecture (skip) | — |
| Table I / metrics | bar paper vs ours | `outputs/figures/market_metrics.svg` |
| CMC ranks | line | `outputs/figures/cmc.svg` |
| Train dynamics | loss curves | `outputs/figures/train_loss.svg` |

## Datasets
| Dataset | Role | Local |
|---------|------|-------|
| Market-1501 | train + eval | `projects/datasets/market1501` (~270 MiB) |
| DukeMTMC / MSMT17 / Occluded-Duke | paper eval | **missing** → DEVIATIONS |

## Size gate
Market-1501 &lt; 5 GiB → **full train required** after smoke (`stage1=120`, `stage2=60`).

## Upstream
https://github.com/Nirmala891/Context-aware-prompt-based-person-re-identification (CLIP-ReID fork + notebooks). CapReID OT pooling reimplemented from the paper equations into `capreid/`.
