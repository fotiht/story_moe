"""Render the KV-cache figure in README.md from results/bench_*.json.

    python docs/make_figures.py

Writes docs/cache-overhead-light.svg and docs/cache-overhead-dark.svg. Two files
rather than one because GitHub serves the README on a light or a dark page and
picks between them with prefers-color-scheme. The dark version is stepped for
the dark surface rather than being an inverted copy of the light one.

Nothing here computes a number. Every value plotted is read straight out of the
benchmark JSON, so editing this script cannot change what the figure claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Matplotlib writes SVG text as vector outlines by default, and that is left
# alone here. Real <text> would be smaller and selectable, but it would also
# render in whatever font the viewer happens to have, and the label positions
# below are tuned against DejaVu metrics closely enough that a substituted font
# could overlap them.

# Categorical slots 1 and 2 from the reference palette, validated for this pair
# against both surfaces: worst all-pairs CVD dE 24.7 light and 26.8 dark.
THEME = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", cached="#2a78d6", uncached="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", cached="#3987e5", uncached="#d95926",
    ),
}


def load(name: str) -> tuple[list[int], list[float], list[float]]:
    """Return (context lengths, cached ms/token, uncached ms/token)."""
    rows = json.loads((RESULTS / name).read_text())["rows"]
    ctx = [r["prompt_tokens"] + r["replay_tokens"] for r in rows]
    return ctx, [r["cached_ms_per_token"] for r in rows], [r["uncached_ms_per_token"] for r in rows]


def draw(mode: str, out: Path) -> None:
    c = THEME[mode]
    series = {
        1: {"dense": load("bench_dense.json"), "moe": load("bench_moe.json")},
        32: {"dense": load("bench_dense_b32.json"), "moe": load("bench_moe_b32.json")},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    fig.patch.set_facecolor(c["surface"])

    for ax, batch in zip(axes, (1, 32)):
        ax.set_facecolor(c["surface"])
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color=c["grid"], linewidth=1, alpha=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
            ax.spines[side].set_linewidth(1)
        ax.tick_params(colors=c["muted"], labelsize=9, length=0)

        for model, style in (("dense", "-"), ("moe", (0, (5, 2)))):
            ctx, cached, uncached = series[batch][model]
            ax.plot(ctx, uncached, linestyle=style, color=c["uncached"],
                    linewidth=2, marker="o", markersize=6,
                    markeredgecolor=c["surface"], markeredgewidth=1.5, zorder=3)
            ax.plot(ctx, cached, linestyle=style, color=c["cached"],
                    linewidth=2, marker="o", markersize=6,
                    markeredgecolor=c["surface"], markeredgewidth=1.5, zorder=3)

        ax.set_xscale("log", base=2)
        ticks = [16, 32, 64, 128, 256, 512]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlim(14, 680)
        ax.set_ylim(0, 29)
        ax.set_title(f"batch {batch}", color=c["ink"], fontsize=11,
                     fontweight="bold", loc="left", pad=10)

    axes[0].set_ylabel("milliseconds per decoded token", color=c["secondary"],
                       fontsize=9.5, labelpad=8)
    fig.supxlabel("context length (prompt tokens + generated tokens)",
                  color=c["secondary"], fontsize=9.5, y=0.045)

    # Identity is carried twice. The left panel gets a legend, the right panel
    # direct labels, so neither rests on color alone.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=c["cached"], linewidth=2, linestyle="-", label="dense, cached"),
        Line2D([], [], color=c["uncached"], linewidth=2, linestyle="-", label="dense, uncached"),
        Line2D([], [], color=c["cached"], linewidth=2, linestyle=(0, (5, 2)), label="MoE, cached"),
        Line2D([], [], color=c["uncached"], linewidth=2, linestyle=(0, (5, 2)), label="MoE, uncached"),
    ]
    leg = axes[0].legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.02), ncols=2,
        frameon=False, fontsize=8.5, labelcolor=c["secondary"],
        handlelength=2.4, columnspacing=1.6, handletextpad=0.6, labelspacing=0.5)
    for t in leg.get_texts():
        t.set_color(c["secondary"])

    ax1, ax32 = axes
    for y, text, color, dy in (
        (series[32]["moe"][2][-1], "MoE uncached", c["uncached"], 0),
        (series[32]["moe"][1][-1], "MoE cached", c["cached"], 9),
        (series[32]["dense"][2][-1], "dense uncached", c["uncached"], -9),
        (series[32]["dense"][1][-1], "dense cached", c["cached"], 0),
    ):
        ax32.annotate(text, (512, y), xytext=(10, dy), textcoords="offset points",
                      color=color, fontsize=8.5, fontweight="bold", va="center",
                      ha="left", zorder=4, annotation_clip=False)

    ax1.annotate("both paths flat, whatever the context:\nthe step is bound by kernel launch",
                 xy=(15.5, 26.2), color=c["secondary"], fontsize=8.5,
                 ha="left", va="top", style="italic")
    ax32.annotate("uncached lifts off once its recomputation\nexceeds the overhead floor",
                  xy=(15.5, 26.2), color=c["secondary"], fontsize=8.5,
                  ha="left", va="top", style="italic")

    fig.suptitle(
        "A KV cache saves nothing until a decode step is compute-bound",
        color=c["ink"], fontsize=13, fontweight="bold", x=0.062, ha="left", y=0.975)
    fig.text(0.062, 0.895,
             "Greedy decode, A100-SXM4-40GB, bf16, median of 5 trials. "
             "Solid dense, dashed MoE. Prefill excluded.",
             color=c["muted"], fontsize=9, ha="left")

    fig.subplots_adjust(left=0.078, right=0.845, top=0.74, bottom=0.155, wspace=0.10)
    fig.savefig(out, format="svg", facecolor=c["surface"])
    plt.close(fig)


if __name__ == "__main__":
    docs = Path(__file__).resolve().parent
    for mode in ("light", "dark"):
        path = docs / f"cache-overhead-{mode}.svg"
        draw(mode, path)
        print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
