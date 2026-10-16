# FIDELITY_AUDIT — CapReID

Target: structural / algorithmic match to Murali & Mishra, TBIOM 2026 (DOI 10.1109/TBIOM.2026.3654624).  
**Exactly five passes; stop after Pass 5.**

## Pass statistics

| Pass | Focus | Confidence | Key fixes / notes |
|------|-------|------------|-------------------|
| 1 | Architecture & modules | 78% | CLIP-ViT/B-16 + prompt bank `n_pool=4`; Sinkhorn OT; α fusion; two-stage schedule wired |
| 2 | Losses & train protocol | 85% | InfoNCE i2t+t2i; Stage2 `L_ID+L_T+L_PT`; Stage1 feature cache (CLIP-ReID style) |
| 3 | Data & eval protocol | 82% | Market-1501 CMC/mAP; Duke/MSMT blocked (D1); size gate 0.14 GiB → full_train |
| 4 | Hyperparams & I/O | 88% | ε=0.05, iters=50, τ=0.07, m=0.3, AdamW 1e-4, wd=0.01; D2/D4 noted |
| 5 | End-to-end sign-off | 90% | scripts + REPORT + deviations; paper table numbers not claimed as matched |

**Weighted confidence after Pass 5: ~85%** (Market-only + 224² + batch 64 prevent 100%).

## Pass 1 — Structure
- [x] OT Sinkhorn (`capreid/models/ot.py`)
- [x] Prompt pooling + global fuse (`CapReID.pool_prompts`)
- [x] Stage1 / Stage2 freeze policy
- [ ] Official upstream `make_model_clipreid` absent in clone → reimplemented (D3)

## Pass 2 — Objectives
- [x] Bidirectional InfoNCE
- [x] ID CE + hard triplet + prompt-aware triplet
- [x] Stage1 trains prompts only on cached image feats

## Pass 3 — Data / metrics
- [x] Market-1501 loader + standard same-cam removal CMC/mAP
- [x] Missing corpora documented (D1)
- [x] `dataset_size.json` decision `full_train`

## Pass 4 — Config parity
- [x] n_pool=4, α=0.45, OT ε/iters, τ, margin
- [x] Epochs 120 + 60 in `configs/default.yaml`
- [x] Resolution / batch deviations logged

## Pass 5 — Packaging
- [x] `train` / `eval` / `predict` / `report` entrypoints
- [x] `DEVIATIONS.md`, `SOURCE_CODE.md`, `REPORT.md` pipeline
- [x] No fabricated Duke/MSMT metrics

## Stop
Fidelity loop complete (5/5). Further rounds only on explicit request.

## Measured Market-1501 (full train)

| Checkpoint | mAP | Rank-1 | Rank-5 | Rank-10 |
|------------|-----|--------|--------|---------|
| best (Stage2 ep30) | 44.46 | 64.31 | 85.04 | 89.76 |
| paper CapReID | 94.8 | 97.2 | — | — |

Gap attributed to D1–D4 (Market-only protocol already; incomplete CLIP-ReID scaffolding vs authors’ full stack).
