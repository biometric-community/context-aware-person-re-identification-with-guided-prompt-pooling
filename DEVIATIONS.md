# DEVIATIONS — CapReID

| ID | Severity | Description |
|----|----------|-------------|
| D1 | High | **DukeMTMC-reID, MSMT17, Occluded-Duke** not available locally. Train/eval/report use **Market-1501 only**. Loaders fail closed if those roots are requested. |
| D2 | Medium | Input resolution **224×224** (open_clip ViT-B-16 openai default) vs paper table footnote **256×128** for several CLIP-ReID baselines. |
| D3 | Medium | Upstream GitHub tree is an incomplete CLIP-ReID fork (missing `make_model_*`); CapReID OT pooling implemented from the paper rather than a line-by-line refactor of missing modules. |
| D4 | Low | Batch size **64** (default) vs paper **128** to fit ~11 GiB GPU; gradient accumulation not used. |
| D5 | Low | Global prompt \(t_g\) approximated as mean over the full prompt bank (paper: mean of class-pooled prompts); functionally similar. |
