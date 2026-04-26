"""Build a compact robustness addendum for the NeurIPS EDTrack paper."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from analysis_utils import summarize_experiment


RESULTS_DIR = Path("results")
MD_PATH = RESULTS_DIR / "edtrack_repair_addendum.md"
CSV_PATH = RESULTS_DIR / "edtrack_repair_addendum.csv"

ROUND_COUNT = 30

ANCHORS = {
    "prior_locked": ("solo_qwen", "solo_qwen_exposed", "homo_qwen"),
    "cue_sensitive": ("baseline", "solo_phi_exposed", "homo_phi"),
    "interaction_reopening": ("solo_ds", "solo_ds_exposed", "homo_ds"),
}

EVALUATION_GROUPS = {
    "mistral": ("solo_mistral", "solo_mistral_exposed", "homo_mistral"),
    "gemma3": ("solo_gemma3_4b", "solo_gemma3_4b_exposed", "homo_gemma3_4b"),
    "gemma4": ("solo_gemma4_e4b", "solo_gemma4_e4b_exposed", "homo_gemma4_e4b"),
    "qwen3_ollama": ("solo_qwen3_ollama", "solo_qwen3_ollama_exposed", "homo_qwen3_ollama"),
    "qwen35_9b": ("solo_qwen35_9b", "solo_qwen35_9b_exposed", "homo_qwen35_9b"),
    "phi3": ("solo_phi3", "solo_phi3_exposed", "homo_phi3"),
    "phi4_ollama": ("solo_phi4_ollama", "solo_phi4_ollama_exposed", "homo_phi4_ollama"),
    "llama3_ollama": ("solo_llama3_ollama", "solo_llama3_ollama_exposed", "homo_llama3_ollama"),
}

ALT_PRIOR_CANDIDATES = {
    "qwen_local": ("solo_qwen", "solo_qwen_exposed", "homo_qwen"),
    "gemma3": ("solo_gemma3_4b", "solo_gemma3_4b_exposed", "homo_gemma3_4b"),
    "qwen3_ollama": ("solo_qwen3_ollama", "solo_qwen3_ollama_exposed", "homo_qwen3_ollama"),
    "llama3": ("solo_llama3_ollama", "solo_llama3_ollama_exposed", "homo_llama3_ollama"),
}


def load_summary_rows() -> dict[str, dict]:
    rows = {}
    with (RESULTS_DIR / "experiment_summary.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["experiment"]] = row
    return rows


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def as_int(row: dict, key: str) -> int:
    return int(float(row[key]))


def triplet_rows(names: tuple[str, str, str]) -> tuple[dict, dict, dict]:
    rows = tuple(summarize_experiment(name) for name in names)
    if any(row is None for row in rows):
        missing = [name for name, row in zip(names, rows) if row is None]
        raise RuntimeError(f"Missing summaries for {missing}")
    return rows  # type: ignore[return-value]


def feature_entropy(rows: tuple[dict, dict, dict]) -> dict[str, float]:
    solo, exposed, homo = rows
    return {
        "solo_entropy": solo["mean_entropy"],
        "exposed_entropy": exposed["mean_entropy"],
        "homo_entropy": homo["mean_entropy"],
    }


def feature_entropy_persistence(rows: tuple[dict, dict, dict]) -> dict[str, float]:
    solo, exposed, homo = rows
    return {
        "solo_entropy": solo["mean_entropy"],
        "exposed_entropy": exposed["mean_entropy"],
        "homo_entropy": homo["mean_entropy"],
        "solo_persistence": solo["mean_action_persistence"],
        "exposed_persistence": exposed["mean_action_persistence"],
        "homo_persistence": homo["mean_action_persistence"],
    }


def feature_entropy_jsd(rows: tuple[dict, dict, dict]) -> dict[str, float]:
    solo, exposed, homo = rows
    return {
        "solo_entropy": solo["mean_entropy"],
        "exposed_entropy": exposed["mean_entropy"],
        "homo_entropy": homo["mean_entropy"],
        "homo_jsd": homo["mean_policy_divergence"],
    }


FEATURE_BUILDERS = {
    "entropy_only": feature_entropy,
    "entropy_plus_persistence": feature_entropy_persistence,
    "entropy_plus_homo_jsd": feature_entropy_jsd,
}


def euclidean(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(sum((left[key] - right[key]) ** 2 for key in left.keys()))


def classify(features: dict[str, float], anchors: dict[str, dict[str, float]]) -> dict[str, float | str]:
    distances = {label: euclidean(features, anchor) for label, anchor in anchors.items()}
    ranked = sorted(distances.items(), key=lambda item: item[1])
    best_label, best_distance = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
    return {
        "label": best_label,
        "distance": best_distance,
        "margin": second_distance - best_distance,
    }


def build_classifier_robustness() -> dict:
    anchor_triplets = {label: triplet_rows(names) for label, names in ANCHORS.items()}
    results: dict[str, dict] = {"feature_sets": {}, "anchor_sensitivity": {}}

    baseline_labels = {}
    for feature_name, builder in FEATURE_BUILDERS.items():
        anchors = {label: builder(rows) for label, rows in anchor_triplets.items()}
        family_rows = {}
        for family, names in EVALUATION_GROUPS.items():
            rows = triplet_rows(names)
            pred = classify(builder(rows), anchors)
            family_rows[family] = pred
            if feature_name == "entropy_only":
                baseline_labels[family] = pred["label"]
        results["feature_sets"][feature_name] = family_rows

    agreement = {}
    for feature_name, family_rows in results["feature_sets"].items():
        if feature_name == "entropy_only":
            continue
        matches = sum(1 for family, pred in family_rows.items() if pred["label"] == baseline_labels[family])
        agreement[feature_name] = {"matches": matches, "total": len(family_rows)}
    results["feature_agreement"] = agreement

    cue_anchor = triplet_rows(ANCHORS["cue_sensitive"])
    interact_anchor = triplet_rows(ANCHORS["interaction_reopening"])
    for alt_name, names in ALT_PRIOR_CANDIDATES.items():
        anchors = {
            "prior_locked": feature_entropy(triplet_rows(names)),
            "cue_sensitive": feature_entropy(cue_anchor),
            "interaction_reopening": feature_entropy(interact_anchor),
        }
        flips = 0
        family_rows = {}
        for family, triplet in EVALUATION_GROUPS.items():
            pred = classify(feature_entropy(triplet_rows(triplet)), anchors)
            family_rows[family] = pred
            if pred["label"] != baseline_labels[family]:
                flips += 1
        results["anchor_sensitivity"][alt_name] = {
            "label_flips": flips,
            "total": len(family_rows),
            "families": family_rows,
        }

    return results


def build_backend_rows(summary_rows: dict[str, dict]) -> list[dict]:
    names = [
        "baseline",
        "single_norm",
        "single_norm_temp07",
        "single_norm_matched",
        "multi_free",
        "multi_norm",
        "temp_low",
        "temp_mid",
        "temp_high",
        "multi_norm_temp07",
        "multi_free_matched",
        "multi_norm_matched",
        "multi_free_rotated",
        "multi_norm_rotated",
        "multi_free_phi3_qwen3_ds_ollama",
        "multi_norm_phi3_qwen3_ds_ollama",
        "multi_free_phi4_qwen35_ds_ollama",
        "multi_norm_phi4_qwen35_ds_ollama",
        "multi_free_phi4_llama3_ds_ollama",
        "multi_norm_phi4_llama3_ds_ollama",
        "solo_qwen",
        "solo_qwen3_ollama",
        "solo_phi4_ollama",
    ]
    rows = []
    for name in names:
        row = summary_rows[name]
        rows.append(
            {
                "experiment": name,
                "episodes": as_int(row, "num_episodes"),
                "entropy": round(as_float(row, "mean_entropy"), 3),
                "persistence": round(as_float(row, "mean_action_persistence"), 3),
                "jsd": round(as_float(row, "mean_policy_divergence"), 3),
                "parser_failures": as_int(row, "parser_failures"),
            }
        )
    return rows


def build_parser_rows(summary_rows: dict[str, dict]) -> list[dict]:
    nonzero = []
    for row in summary_rows.values():
        failures = as_int(row, "parser_failures")
        if failures <= 0:
            continue
        total_actions = as_int(row, "num_episodes") * as_int(row, "num_agents") * ROUND_COUNT
        nonzero.append(
            {
                "experiment": row["experiment"],
                "episodes": as_int(row, "num_episodes"),
                "parser_failures": failures,
                "failure_rate": failures / total_actions,
                "entropy": as_float(row, "mean_entropy"),
            }
        )
    nonzero.sort(key=lambda item: (-item["parser_failures"], item["experiment"]))
    return nonzero


def build_csv_rows(backend_rows: list[dict], parser_rows: list[dict], classifier: dict) -> list[dict]:
    rows = []
    for row in backend_rows:
        rows.append({"section": "backend_decode", **row})
    for row in parser_rows:
        rows.append({"section": "parser_failures", **row})
    for feature_name, family_rows in classifier["feature_sets"].items():
        for family, pred in family_rows.items():
            rows.append(
                {
                    "section": "classifier_feature_set",
                    "feature_set": feature_name,
                    "family": family,
                    "label": pred["label"],
                    "margin": round(pred["margin"], 6),
                }
            )
    for alt_name, result in classifier["anchor_sensitivity"].items():
        rows.append(
            {
                "section": "anchor_sensitivity",
                "anchor_variant": alt_name,
                "label_flips": result["label_flips"],
                "total": result["total"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(backend_rows: list[dict], parser_rows: list[dict], classifier: dict) -> str:
    backend_lookup = {row["experiment"]: row for row in backend_rows}
    lines = [
        "# EDTrack Repair Addendum",
        "",
            "This addendum compacts the last-round review fixes into four narrow checks: backend/decode sensitivity, triad-composition robustness, parser-failure concentration, and nearest-anchor classifier stability.",
        "",
        "## 1. Backend and decode sensitivity",
        "",
        "| Condition | Episodes | Mean H | Persistence | JSD | Parser failures |",
        "|-----------|----------|--------|-------------|-----|-----------------|",
    ]
    for row in backend_rows:
        lines.append(
            f"| {row['experiment']} | {row['episodes']} | {row['entropy']:.3f} | "
            f"{row['persistence']:.3f} | {row['jsd']:.3f} | {row['parser_failures']} |"
        )

    lines.extend(
        [
            "",
            "Reading:",
            "",
            "- The local single-agent norm cell is also stack- and temperature-sensitive: `single_norm` at the original 0.5 temperature remains the violating-attractor cell (`H=0.046`), while `single_norm_temp07` and `single_norm_matched` both collapse to deterministic cooperation.",
            "- `multi_free` / `multi_norm` remain broad on the original local stack (`0.975` / `1.006`).",
            f"- Matched-temperature local triad checks at 0.7 are low-entropy (`temp_low`, `temp_mid`, `temp_high` at `H={backend_lookup['temp_low']['entropy']:.3f}`, `{backend_lookup['temp_mid']['entropy']:.3f}`, `{backend_lookup['temp_high']['entropy']:.3f}`; `multi_norm_temp07` at `H={backend_lookup['multi_norm_temp07']['entropy']:.3f}`), so this addendum is evidence of decoding/backend sensitivity, not region invariance.",
            f"- The matched-backend Ollama triad moves into a near-zero support-concentration regime in both `multi_free_matched` and `multi_norm_matched` (`H={backend_lookup['multi_free_matched']['entropy']:.3f}` and `H={backend_lookup['multi_norm_matched']['entropy']:.3f}`), so the core heterogeneous result is not backend-invariant.",
            f"- Within that matched-backend setting, rotating agent order and neutralizing the role-like IDs leaves the regime similarly concentrated (`multi_free_rotated`, `multi_norm_rotated` at `H={backend_lookup['multi_free_rotated']['entropy']:.3f}` and `H={backend_lookup['multi_norm_rotated']['entropy']:.3f}`), suggesting that backend choice matters more than ID order in this robustness probe.",
            f"- Three additional all-Ollama heterogeneous compositions also remain in a low-support band: Phi3/Qwen3/DeepSeek (`H={backend_lookup['multi_free_phi3_qwen3_ds_ollama']['entropy']:.3f}` / `{backend_lookup['multi_norm_phi3_qwen3_ds_ollama']['entropy']:.3f}`), Phi4/Qwen3.5/DeepSeek (`H={backend_lookup['multi_free_phi4_qwen35_ds_ollama']['entropy']:.3f}` / `{backend_lookup['multi_norm_phi4_qwen35_ds_ollama']['entropy']:.3f}`), and Phi4/Llama3/DeepSeek (`H={backend_lookup['multi_free_phi4_llama3_ds_ollama']['entropy']:.3f}` / `{backend_lookup['multi_norm_phi4_llama3_ds_ollama']['entropy']:.3f}`). This strengthens the claim that the matched-backend collapse is not a simple role-order artifact, while still remaining stack-conditioned.",
            "- The existing family-line controls remain the cleanest within-family backend check: Qwen stays prior-locked on both stacks, while Phi shifts sharply between local `llama.cpp` and Ollama.",
            "",
            "## 2. Parser-failure concentration",
            "",
            "| Condition | Episodes | Parser failures | Failure rate / action | Mean H |",
            "|-----------|----------|-----------------|-----------------------|--------|",
        ]
    )
    for row in parser_rows:
        lines.append(
            f"| {row['experiment']} | {row['episodes']} | {row['parser_failures']} | "
            f"{row['failure_rate']:.4%} | {row['entropy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Reading:",
            "",
            "- Parser failures remain concentrated in the temperature pilot, especially `temp_high`, rather than in the main benchmark conditions.",
            "- The new matched-backend robustness runs have zero parser failures, so they isolate backend/condition effects without introducing extra parse noise.",
            "- The temperature pilot should therefore be read as a small diagnostic check rather than as evidence of temperature-invariant response regions.",
            "",
            "## 3. Nearest-anchor classifier robustness",
            "",
            "| Family | Entropy only | Entropy + persistence | Entropy + homo-JSD |",
            "|--------|--------------|-----------------------|--------------------|",
        ]
    )
    entropy_only = classifier["feature_sets"]["entropy_only"]
    entropy_persistence = classifier["feature_sets"]["entropy_plus_persistence"]
    entropy_jsd = classifier["feature_sets"]["entropy_plus_homo_jsd"]
    for family in entropy_only:
        lines.append(
            f"| {family} | {entropy_only[family]['label']} | "
            f"{entropy_persistence[family]['label']} | {entropy_jsd[family]['label']} |"
        )

    lines.extend(
        [
            "",
            "Feature-set agreement against entropy-only labels:",
            "",
        ]
    )
    for feature_name, stats in classifier["feature_agreement"].items():
        lines.append(f"- `{feature_name}`: {stats['matches']}/{stats['total']} label matches.")

    lines.extend(
        [
            "",
            "Alternative prior-anchor sensitivity (swapping the prior-locked anchor among exact-zero prior families):",
            "",
        ]
    )
    for anchor_name, result in classifier["anchor_sensitivity"].items():
        lines.append(f"- `{anchor_name}`: {result['label_flips']}/{result['total']} label flips.")

    lines.extend(
        [
            "",
            "Overall reading:",
            "",
            "- The nearest-anchor rule remains intentionally simple, but these checks show it is not purely ad hoc: adding persistence or homogeneous JSD does not materially change the held-out labels in the current set.",
            "- Sensitivity on the prior-locked side is especially low because multiple held-out families share the same exact zero corner.",
        ]
    )

    return "\n".join(lines) + "\n"


def main():
    summary_rows = load_summary_rows()
    backend_rows = build_backend_rows(summary_rows)
    parser_rows = build_parser_rows(summary_rows)
    classifier = build_classifier_robustness()
    write_csv(CSV_PATH, build_csv_rows(backend_rows, parser_rows, classifier))
    MD_PATH.write_text(build_markdown(backend_rows, parser_rows, classifier), encoding="utf-8")
    print(f"Wrote {MD_PATH} and {CSV_PATH}")


if __name__ == "__main__":
    main()
