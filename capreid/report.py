"""Generate CapReID figures (dev-plot) and REPORT.md from real logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from . import plot_style


def _load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logs", default="outputs/logs")
    ap.add_argument("--figdir", default="outputs/figures")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logs = Path(args.logs)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    plot_style.apply_rcparams()

    train = _load_json(logs / "train_summary.json") or {}
    eval_r = _load_json(logs / "eval_results.json") or {}
    history = train.get("history") or []
    metrics = eval_r.get("metrics") or train.get("final_metrics") or {}

    # Loss curves
    s1 = [h for h in history if h.get("stage") == 1]
    s2 = [h for h in history if h.get("stage") == 2]
    fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
    if s1:
        ax.plot(
            [h["epoch"] for h in s1],
            [h["loss"] for h in s1],
            color=plot_style.MULTI_SERIES_COLORS[0],
            linewidth=plot_style.LINEWIDTH_MAIN,
            label="Stage 1 (prompts)",
        )
    if s2:
        ax.plot(
            [h["epoch"] for h in s2],
            [h["loss"] for h in s2],
            color=plot_style.MULTI_SERIES_COLORS[1],
            linewidth=plot_style.LINEWIDTH_MAIN,
            label="Stage 2 (image)",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CapReID training loss (Market-1501)")
    ax.legend()
    ax.grid(True, alpha=plot_style.GRID_ALPHA)
    plot_style.apply_style(ax)
    loss_svg = figdir / "train_loss.svg"
    fig.savefig(loss_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    # Metric bars vs paper Market numbers (Table I CapReID)
    paper_market = {"mAP": 94.8, "Rank-1": 97.2}
    ours = {
        "mAP": float(metrics.get("mAP", 0.0)),
        "Rank-1": float(metrics.get("Rank-1", 0.0)),
    }
    fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
    keys = ["mAP", "Rank-1"]
    x = range(len(keys))
    w = 0.35
    ax.bar(
        [i - w / 2 for i in x],
        [paper_market[k] for k in keys],
        width=w,
        color=plot_style.MULTI_SERIES_COLORS[0],
        label="Paper CapReID (Market)",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [ours[k] for k in keys],
        width=w,
        color=plot_style.MULTI_SERIES_COLORS[1],
        label="Ours (Market)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Market-1501: paper vs ours")
    ax.legend()
    ax.grid(True, axis="y", alpha=plot_style.GRID_ALPHA)
    plot_style.apply_style(ax)
    bar_svg = figdir / "market_metrics.svg"
    fig.savefig(bar_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    # CMC-style from eval ranks if present
    ranks = []
    for r in cfg.get("eval", {}).get("ranks", [1, 5, 10]):
        key = f"Rank-{r}"
        if key in metrics:
            ranks.append((r, float(metrics[key])))
    cmc_svg = None
    if ranks:
        fig, ax = plt.subplots(figsize=plot_style.FIGSIZE)
        ax.plot(
            [r for r, _ in ranks],
            [v for _, v in ranks],
            color=plot_style.BAR_COLOR,
            linewidth=plot_style.LINEWIDTH_MAIN,
            marker="o",
            markersize=8,
        )
        ax.set_xlabel("Rank")
        ax.set_ylabel("CMC (%)")
        ax.set_title("Market-1501 CMC (ours)")
        ax.grid(True, alpha=plot_style.GRID_ALPHA)
        plot_style.apply_style(ax)
        cmc_svg = figdir / "cmc.svg"
        fig.savefig(cmc_svg, format="svg", bbox_inches="tight")
        plt.close(fig)

    # REPORT.md
    blocked = not metrics
    lines = [
        "# REPORT — CapReID",
        "",
        f"**Paper:** {cfg.get('paper', {}).get('title', 'CapReID')}",
        f"**DOI:** {cfg.get('paper', {}).get('doi', '')}",
        "",
        "## Summary",
        "",
    ]
    if blocked:
        lines += [
            "**BLOCKED / incomplete:** no eval metrics found under `outputs/logs/`.",
            "Run `scripts/train.sh` / `scripts/eval.sh` on Market-1501 first.",
            "",
        ]
    else:
        lines += [
            "| Split | mAP | Rank-1 | Rank-5 | Rank-10 |",
            "|-------|-----|--------|--------|---------|",
            (
                f"| Market-1501 (ours) | {ours['mAP']:.2f} | {ours['Rank-1']:.2f} | "
                f"{float(metrics.get('Rank-5', 0)):.2f} | {float(metrics.get('Rank-10', 0)):.2f} |"
            ),
            f"| Market-1501 (paper CapReID) | {paper_market['mAP']:.1f} | {paper_market['Rank-1']:.1f} | — | — |",
            "",
            "DukeMTMC / MSMT17 / Occluded-Duke were **not** run (data unavailable locally); see `DEVIATIONS.md`.",
            "",
            "## Figures",
            "",
            "### Training loss",
            "",
            f"![train loss]({loss_svg.as_posix()})",
            "",
            "### Market metrics vs paper",
            "",
            f"![metrics]({bar_svg.as_posix()})",
            "",
        ]
        if cmc_svg:
            lines += ["### CMC", "", f"![cmc]({cmc_svg.as_posix()})", ""]
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/setup_env.sh",
        "bash scripts/train_full.sh",
        "bash scripts/eval.sh",
        "bash scripts/report.sh",
        "```",
        "",
    ]
    Path("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote REPORT.md and figures under {figdir}")


if __name__ == "__main__":
    main()
