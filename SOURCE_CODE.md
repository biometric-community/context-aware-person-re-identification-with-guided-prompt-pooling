# SOURCE_CODE

| Field | Value |
|-------|-------|
| Upstream URL | https://github.com/Nirmala891/Context-aware-prompt-based-person-re-identification |
| License (upstream) | Not clearly stated in README at clone time |
| Relationship | Authors’ CapReID / CLIP-ReID scaffolding |

## What we reused
- Market-1501 filename parsing and CMC/mAP protocol (aligned with standard ReID / prior 2D-SNet rebuild).
- Two-stage CLIP-ReID training *concept* (prompt stage then image fine-tune) from upstream processors.
- open_clip `ViT-B-16` / `openai` weights instead of vendoring incomplete upstream `model/` tree (clone lacked `make_model_clipreid.py`).

## What we reimplemented
- Sinkhorn OT prompt pooling (`capreid/models/ot.py`) per paper ε=0.05 / 50 iters.
- Prompt bank + α fusion of class-pooled and global prompts.
- Bidirectional InfoNCE, ID CE, hard triplet, prompt-aware triplet (`capreid/losses.py`).
- Project layout: `python -m capreid.{train,eval,predict,report}` + `scripts/*.sh`.

## Local clone (not shipped)
`upstream/` is gitignored; do not vendor the full CLIP-ReID tree into the published repo.
