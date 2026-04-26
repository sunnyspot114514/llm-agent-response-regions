"""Build a compact overview figure for the response framework."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


matplotlib.use("Agg")

RESULTS_DIR = Path("results")


def main():
    fig, ax = plt.subplots(figsize=(14, 4.05))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    column_x = [0.055, 0.385, 0.715]
    card_w = 0.225
    card_h = 0.105
    columns = [
        (column_x[0], "Evaluation assumptions", "#5B7EA4", "#f4f7fb"),
        (column_x[1], "Observed response labels", "#8A78A8", "#f7f5fb"),
        (column_x[2], "Evidence types", "#8B7C5E", "#fbf8f0"),
    ]

    for x, title, color, _fill in columns:
        ax.text(x, 0.805, title, ha="left", va="center", fontsize=11.0, fontweight="bold", color=color)
        ax.plot([x, x + card_w], [0.775, 0.775], color=color, linewidth=2.0, solid_capstyle="round")

    cards = [
        (column_x[0], 0.635, "Model priors", "isolated support; line stability", "#f4f7fb", "#5B7EA4"),
        (column_x[0], 0.465, "Information structure", "visibility; transcript cues", "#f4f7fb", "#5B7EA4"),
        (column_x[0], 0.295, "Norm binding", "mode; target semantics", "#f4f7fb", "#5B7EA4"),
        (column_x[1], 0.635, "Prior-locked", "low support at baseline", "#f7f5fb", "#8A78A8"),
        (column_x[1], 0.465, "Cue-sensitive", "scripted exposure compresses", "#f7f5fb", "#8A78A8"),
        (column_x[1], 0.295, "Interaction reopening", "play restores support", "#f7f5fb", "#8A78A8"),
        (column_x[1], 0.125, "Deployment-sensitive", "backend changes regime", "#f7f5fb", "#8A78A8"),
        (column_x[2], 0.635, "Metrics", "entropy, persistence, JSD", "#fbf8f0", "#8B7C5E"),
        (column_x[2], 0.465, "Forecasts", "held-out; cross-generation", "#fbf8f0", "#8B7C5E"),
        (column_x[2], 0.295, "Mechanism", "late residual + MLP anchor", "#fbf8f0", "#8B7C5E"),
    ]

    def card(x, y, title, body, fill, edge):
        patch = FancyBboxPatch(
            (x, y),
            card_w,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.05,
        )
        ax.add_patch(patch)
        ax.text(x + 0.014, y + 0.069, title, ha="left", va="center", fontsize=9.1, fontweight="bold", color="#202020")
        ax.text(x + 0.014, y + 0.033, body, ha="left", va="center", fontsize=7.8, color="#333333")

    for args in cards:
        card(*args)

    arrows = [
        ((column_x[0] + card_w + 0.018, 0.688), (column_x[1] - 0.018, 0.688)),
        ((column_x[0] + card_w + 0.018, 0.518), (column_x[1] - 0.018, 0.518)),
        ((column_x[0] + card_w + 0.018, 0.348), (column_x[1] - 0.018, 0.348)),
        ((column_x[1] + card_w + 0.018, 0.688), (column_x[2] - 0.018, 0.688)),
        ((column_x[1] + card_w + 0.018, 0.518), (column_x[2] - 0.018, 0.518)),
        ((column_x[1] + card_w + 0.018, 0.348), (column_x[2] - 0.018, 0.348)),
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.25, color="#8b949e")
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.925,
        "Controlled evaluation of behavioral support concentration",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        0.5,
        0.045,
        "Evaluative warning: support concentration may reflect prior rigidity, cue-induced compression, interaction reopening, or serving-stack sensitivity.",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#374151",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.06)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / "response_framework_overview.png", dpi=320, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "response_framework_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote results/response_framework_overview.png and .pdf")


if __name__ == "__main__":
    main()
