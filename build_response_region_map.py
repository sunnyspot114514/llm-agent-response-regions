from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "experiment_summary.csv"
OUT_PATH = ROOT / "results" / "response_region_map.png"
OUT_PDF_PATH = ROOT / "results" / "response_region_map.pdf"


FAMILIES = [
    ("Qwen3-8B", "solo_qwen", "solo_qwen_exposed", "homo_qwen"),
    ("Phi-4-mini (local)", "baseline", "solo_phi_exposed", "homo_phi"),
    ("DeepSeek-R1", "solo_ds", "solo_ds_exposed", "homo_ds"),
    ("Mistral-7B", "solo_mistral", "solo_mistral_exposed", "homo_mistral"),
    ("Llama3-8B", "solo_llama3_ollama", "solo_llama3_ollama_exposed", "homo_llama3_ollama"),
    ("Gemma-3 4B", "solo_gemma3_4b", "solo_gemma3_4b_exposed", "homo_gemma3_4b"),
    ("Gemma-4 E4B", "solo_gemma4_e4b", "solo_gemma4_e4b_exposed", "homo_gemma4_e4b"),
    ("Qwen3.5-9B", "solo_qwen35_9b", "solo_qwen35_9b_exposed", "homo_qwen35_9b"),
    ("Phi3-3.8B", "solo_phi3", "solo_phi3_exposed", "homo_phi3"),
    ("Phi-4-mini (Ollama)", "solo_phi4_ollama", "solo_phi4_ollama_exposed", "homo_phi4_ollama"),
]

LABEL_LAYOUT = {
    "Qwen3-8B": {"x": 0.07, "y": 0.025, "ha": "left", "va": "bottom"},
    "Mistral-7B": {"x": -0.10, "y": 0.34, "ha": "right", "va": "bottom"},
    "Llama3-8B": {"x": -0.19, "y": 0.03, "ha": "right", "va": "bottom"},
    "Gemma-3 4B": {"x": -0.17, "y": -0.06, "ha": "right", "va": "top"},
    "Gemma-4 E4B": {"x": 0.17, "y": -0.085, "ha": "left", "va": "top"},
    "Qwen3.5-9B": {"x": 0.16, "y": 0.11, "ha": "left", "va": "bottom"},
    "Phi3-3.8B": {"x": 0.24, "y": 0.02, "ha": "left", "va": "bottom"},
    "Phi-4-mini (Ollama)": {"x": 0.18, "y": 0.215, "ha": "left", "va": "bottom"},
    "Phi-4-mini (local)": {"dx": 0.03, "dy": 0.03, "ha": "left", "va": "bottom"},
    "DeepSeek-R1": {"dx": 0.03, "dy": 0.03, "ha": "left", "va": "bottom"},
}


def load_entropies() -> dict[str, float]:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["experiment"]: float(row["mean_entropy"]) for row in reader}


def main() -> None:
    entropies = load_entropies()

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.cm.viridis

    available_families = [
        row for row in FAMILIES if all(key in entropies for key in row[1:])
    ]
    if not available_families:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "No complete solo/scripted/homogeneous triplets\navailable in the current subset export.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.text(
            0.5,
            0.35,
            "Use the bundled canonical results/experiment_summary.csv\nfor the full paper figure.",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )
        fig.tight_layout()
        fig.savefig(OUT_PATH, dpi=260)
        fig.savefig(OUT_PDF_PATH)
        return

    exposed_values = [entropies[e] for _, _, e, _ in available_families]
    vmin, vmax = min(exposed_values), max(exposed_values)

    for label, solo_key, exposed_key, homo_key in available_families:
        x = entropies[solo_key]
        y = entropies[homo_key]
        c = entropies[exposed_key]
        ax.scatter(
            x,
            y,
            s=140,
            c=[c],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        layout = LABEL_LAYOUT.get(
            label,
            {"dx": 0.025, "dy": 0.02, "ha": "left", "va": "bottom"},
        )
        if "x" in layout and "y" in layout:
            text_x = layout["x"]
            text_y = layout["y"]
            dx = text_x - x
            dy = text_y - y
        else:
            dx = layout["dx"]
            dy = layout["dy"]
            text_x = x + dx
            text_y = y + dy
        use_arrow = abs(dx) > 0.06 or abs(dy) > 0.06
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(text_x, text_y),
            textcoords="data",
            fontsize=8.5,
            ha=layout["ha"],
            va=layout["va"],
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#dddddd",
                "boxstyle": "round,pad=0.18",
            },
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "color": "#777777",
                    "lw": 0.8,
                    "shrinkA": 4,
                    "shrinkB": 4,
                }
                if use_arrow
                else None
            ),
            zorder=4,
        )

    ax.axvspan(-0.05, 0.18, ymin=0.0, ymax=0.15, color="#d8f0d2", alpha=0.8, zorder=0)
    ax.text(0.005, 0.075, "prior-\nlocked", fontsize=10.5, weight="bold", color="#255d22")

    ax.axvspan(1.1, 1.55, ymin=0.45, ymax=0.95, color="#f6e3b7", alpha=0.8, zorder=0)
    ax.text(1.12, 1.50, "cue-locked", fontsize=12, weight="bold", color="#8a5a00")

    ax.axvspan(0.72, 1.1, ymin=0.32, ymax=0.78, color="#dce9fb", alpha=0.8, zorder=0)
    ax.text(0.74, 1.02, "interaction-\nunlocked", fontsize=12, weight="bold", color="#1d4e89")

    ax.text(
        0.03,
        1.55,
        "Color = scripted-transcript entropy",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    ax.set_xlim(-0.05, 1.65)
    ax.set_ylim(-0.05, 1.7)
    ax.set_xlabel("Isolated entropy")
    ax.set_ylabel("Homogeneous entropy")
    ax.set_title("Response regions across local, held-out, and cross-generation family probes")
    ax.grid(alpha=0.2, linewidth=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Scripted-transcript entropy")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=260)
    fig.savefig(OUT_PDF_PATH)


if __name__ == "__main__":
    main()
