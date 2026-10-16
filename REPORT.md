# REPORT — CapReID

**Paper:** Context-Aware Person Re-Identification With Guided Prompt Pooling
**DOI:** 10.1109/TBIOM.2026.3654624

## Summary

| Split | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|-----|--------|--------|---------|
| Market-1501 (ours) | 44.46 | 64.31 | 85.04 | 89.76 |
| Market-1501 (paper CapReID) | 94.8 | 97.2 | — | — |

DukeMTMC / MSMT17 / Occluded-Duke were **not** run (data unavailable locally); see `DEVIATIONS.md`.

## Figures

### Training loss

![train loss](outputs/figures/train_loss.svg)

### Market metrics vs paper

![metrics](outputs/figures/market_metrics.svg)

### CMC

![cmc](outputs/figures/cmc.svg)

## Reproduce

```bash
bash scripts/setup_env.sh
bash scripts/train_full.sh
bash scripts/eval.sh
bash scripts/report.sh
```

