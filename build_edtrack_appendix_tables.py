from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path

from analysis_utils import bootstrap_mean_ci, permutation_test_mean, summarize_experiment


RESULTS_DIR = Path("results")
EXPERIMENT_SUMMARY = RESULTS_DIR / "experiment_summary.csv"


def load_experiment_summary() -> dict[str, dict[str, str]]:
    with EXPERIMENT_SUMMARY.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {row["experiment"]: row for row in reader}


def format_float(value: float) -> str:
    return f"{value:.3f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, header: str, lines: list[str]) -> None:
    path.write_text("\n".join([header, "", *lines, ""]) + "\n", encoding="utf-8")


def bootstrap_delta_ci(left: list[float], right: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    if not left or not right:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    observed = (sum(right) / len(right)) - (sum(left) / len(left))
    samples: list[float] = []
    for _ in range(n_boot):
        left_sample = [left[rng.randrange(len(left))] for _ in range(len(left))]
        right_sample = [right[rng.randrange(len(right))] for _ in range(len(right))]
        samples.append((sum(right_sample) / len(right_sample)) - (sum(left_sample) / len(left_sample)))
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return observed, low, high


def cliffs_delta(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    greater = 0
    less = 0
    for lval in left:
        for rval in right:
            if rval > lval:
                greater += 1
            elif rval < lval:
                less += 1
    return (greater - less) / (len(left) * len(right))


def fdr_bh(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    m = len(p_values)
    for rank_from_end, (index, pval) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        running = min(running, pval * m / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def build_commons_transfer(summary: dict[str, dict[str, str]]) -> tuple[list[dict], list[str]]:
    families = [
        ("Qwen", "solo_qwen", "solo_qwen_exposed", "homo_qwen", "commons_solo_qwen", "commons_solo_qwen_exposed", "commons_homo_qwen"),
        ("Phi", "baseline", "solo_phi_exposed", "homo_phi", "commons_solo_phi", "commons_solo_phi_exposed", "commons_homo_phi"),
        ("DeepSeek", "solo_ds", "solo_ds_exposed", "homo_ds", "commons_solo_ds", "commons_solo_ds_exposed", "commons_homo_ds"),
    ]
    rows: list[dict] = []
    md_lines = [
        "| Family | Social Solo | Social Scripted | Social Homo | Commons Solo | Commons Scripted | Commons Homo |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, s_solo, s_scripted, s_homo, c_solo, c_scripted, c_homo in families:
        required = [s_solo, s_scripted, s_homo, c_solo, c_scripted, c_homo]
        if any(name not in summary for name in required):
            continue
        row = {
            "family": family,
            "social_solo_entropy": format_float(float(summary[s_solo]["mean_entropy"])),
            "social_scripted_entropy": format_float(float(summary[s_scripted]["mean_entropy"])),
            "social_homo_entropy": format_float(float(summary[s_homo]["mean_entropy"])),
            "commons_solo_entropy": format_float(float(summary[c_solo]["mean_entropy"])),
            "commons_scripted_entropy": format_float(float(summary[c_scripted]["mean_entropy"])),
            "commons_homo_entropy": format_float(float(summary[c_homo]["mean_entropy"])),
        }
        rows.append(row)
        md_lines.append(
            f"| {family} | {row['social_solo_entropy']} | {row['social_scripted_entropy']} | "
            f"{row['social_homo_entropy']} | {row['commons_solo_entropy']} | "
            f"{row['commons_scripted_entropy']} | {row['commons_homo_entropy']} |"
        )
    if not rows:
        md_lines.append("| subset run | missing full commons/social triplets | - | - | - | - | - |")
    return rows, md_lines


def build_auxiliary_metrics() -> tuple[list[dict], list[str]]:
    experiments = [
        "baseline",
        "single_norm",
        "multi_free",
        "multi_norm",
        "multi_free_blind",
        "multi_free_agg",
        "multi_norm_strong",
        "multi_norm_mask",
        "multi_norm_multi",
        "temp_high",
    ]
    rows: list[dict] = []
    md_lines = [
        "| Condition | Mean H | Transition H | Persistence | Between-episode SD | Parser fails |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in experiments:
        summary = summarize_experiment(experiment)
        if summary is None:
            continue
        episode_values = summary["episode_mean_entropies"]
        between_episode_sd = statistics.stdev(episode_values) if len(episode_values) > 1 else 0.0
        row = {
            "condition": experiment,
            "mean_entropy": format_float(summary["mean_entropy"]),
            "mean_transition_entropy": format_float(summary["mean_transition_entropy"]),
            "mean_action_persistence": format_float(summary["mean_action_persistence"]),
            "between_episode_sd": format_float(between_episode_sd),
            "parser_failures": str(summary["parser_failures"]),
        }
        rows.append(row)
        md_lines.append(
            f"| `{experiment}` | {row['mean_entropy']} | {row['mean_transition_entropy']} | "
            f"{row['mean_action_persistence']} | {row['between_episode_sd']} | {row['parser_failures']} |"
        )
    return rows, md_lines


def build_effect_sizes() -> tuple[list[dict], list[str]]:
    contrasts = [
        ("baseline", "single_norm", "single-agent soft norm"),
        ("multi_free", "multi_norm", "soft norm in heterogeneous triad"),
        ("multi_free", "multi_free_blind", "blind visibility"),
        ("multi_free", "multi_free_agg", "aggregate visibility"),
        ("multi_norm", "multi_norm_strong", "strong norm binding"),
        ("multi_norm", "multi_norm_mask", "hard mask binding"),
        ("multi_norm", "multi_norm_multi", "multi-rule binding"),
        ("baseline", "solo_phi_exposed", "scripted transcript on Phi"),
        ("solo_ds", "homo_ds", "DeepSeek interaction reopening"),
    ]
    raw_rows: list[dict] = []
    p_values: list[float] = []
    for left_name, right_name, label in contrasts:
        left_summary = summarize_experiment(left_name)
        right_summary = summarize_experiment(right_name)
        if left_summary is None or right_summary is None:
            continue
        left_values = left_summary["episode_mean_entropies"]
        right_values = right_summary["episode_mean_entropies"]
        left_mean, _, _ = bootstrap_mean_ci(left_values)
        right_mean, _, _ = bootstrap_mean_ci(right_values)
        delta, delta_low, delta_high = bootstrap_delta_ci(left_values, right_values)
        p_value = permutation_test_mean(left_values, right_values)
        p_values.append(p_value)
        raw_rows.append(
            {
                "contrast": label,
                "left": left_name,
                "right": right_name,
                "left_mean_entropy": left_mean,
                "right_mean_entropy": right_mean,
                "delta_entropy": delta,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "cliffs_delta": cliffs_delta(left_values, right_values),
                "permutation_p": p_value,
            }
        )

    q_values = fdr_bh(p_values)
    rows: list[dict] = []
    md_lines = [
        "| Contrast | Delta H [95% CI] | Cliff's delta | p | FDR q |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row, q_value in zip(raw_rows, q_values):
        formatted = {
            "contrast": row["contrast"],
            "left": row["left"],
            "right": row["right"],
            "left_mean_entropy": format_float(row["left_mean_entropy"]),
            "right_mean_entropy": format_float(row["right_mean_entropy"]),
            "delta_entropy": format_float(row["delta_entropy"]),
            "delta_ci_low": format_float(row["delta_ci_low"]),
            "delta_ci_high": format_float(row["delta_ci_high"]),
            "cliffs_delta": format_float(row["cliffs_delta"]),
            "permutation_p": f"{row['permutation_p']:.4f}",
            "fdr_q": f"{q_value:.4f}",
        }
        rows.append(formatted)
        md_lines.append(
            f"| {row['contrast']} | {formatted['delta_entropy']} "
            f"[{formatted['delta_ci_low']}, {formatted['delta_ci_high']}] | "
            f"{formatted['cliffs_delta']} | {formatted['permutation_p']} | {formatted['fdr_q']} |"
        )
    if not rows:
        md_lines.append("| subset run | - | - | - | - |")
    return rows, md_lines


def entropy(actions: list[str]) -> float:
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    total = len(actions)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def persistence(actions: list[str]) -> float:
    if len(actions) < 2:
        return 0.0
    same = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    return same / (len(actions) - 1)


def simulate_reference_policies(num_episodes: int = 100, horizon: int = 30, seed: int = 42) -> tuple[list[dict], list[str]]:
    actions = ["cooperate", "defect", "defend", "negotiate", "abstain"]
    rng = random.Random(seed)

    def always_cooperate() -> list[str]:
        return ["cooperate"] * horizon

    def uniform_random() -> list[str]:
        return [rng.choice(actions) for _ in range(horizon)]

    def sticky_markov(p_stay: float = 0.85) -> list[str]:
        seq = [rng.choice(actions)]
        for _ in range(1, horizon):
            if rng.random() < p_stay:
                seq.append(seq[-1])
            else:
                others = [a for a in actions if a != seq[-1]]
                seq.append(rng.choice(others))
        return seq

    specs = [
        ("always_cooperate", "deterministic reference", always_cooperate),
        ("uniform_random", "iid uniform over 5 actions", uniform_random),
        ("sticky_markov_p0.85", "uniform start, then stay with p=0.85", sticky_markov),
    ]

    rows: list[dict] = []
    md_lines = [
        "| Policy | Construction | Mean Entropy | Mean Persistence |",
        "| --- | --- | ---: | ---: |",
    ]

    for name, description, fn in specs:
        episode_entropies: list[float] = []
        episode_persistences: list[float] = []
        for _ in range(num_episodes):
            seq = fn()
            episode_entropies.append(entropy(seq))
            episode_persistences.append(persistence(seq))
        mean_entropy = sum(episode_entropies) / len(episode_entropies)
        mean_persistence = sum(episode_persistences) / len(episode_persistences)
        row = {
            "policy": name,
            "construction": description,
            "num_episodes": str(num_episodes),
            "horizon": str(horizon),
            "mean_entropy": format_float(mean_entropy),
            "mean_persistence": format_float(mean_persistence),
        }
        rows.append(row)
        md_lines.append(
            f"| {name} | {description} | {row['mean_entropy']} | {row['mean_persistence']} |"
        )
    return rows, md_lines


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    summary = load_experiment_summary()

    commons_rows, commons_md = build_commons_transfer(summary)
    write_csv(
        RESULTS_DIR / "commons_transfer_summary.csv",
        commons_rows,
        [
            "family",
            "social_solo_entropy",
            "social_scripted_entropy",
            "social_homo_entropy",
            "commons_solo_entropy",
            "commons_scripted_entropy",
            "commons_homo_entropy",
        ],
    )
    write_markdown(
        RESULTS_DIR / "commons_transfer_summary.md",
        "# Commons Transfer Summary",
        commons_md,
    )

    reference_rows, reference_md = simulate_reference_policies()
    write_csv(
        RESULTS_DIR / "nonllm_reference_policies.csv",
        reference_rows,
        ["policy", "construction", "num_episodes", "horizon", "mean_entropy", "mean_persistence"],
    )
    write_markdown(
        RESULTS_DIR / "nonllm_reference_policies.md",
        "# Non-LLM Reference Policies",
        reference_md,
    )

    auxiliary_rows, auxiliary_md = build_auxiliary_metrics()
    write_csv(
        RESULTS_DIR / "auxiliary_metrics_summary.csv",
        auxiliary_rows,
        [
            "condition",
            "mean_entropy",
            "mean_transition_entropy",
            "mean_action_persistence",
            "between_episode_sd",
            "parser_failures",
        ],
    )
    write_markdown(
        RESULTS_DIR / "auxiliary_metrics_summary.md",
        "# Auxiliary Metrics Summary",
        auxiliary_md,
    )

    effect_rows, effect_md = build_effect_sizes()
    write_csv(
        RESULTS_DIR / "effect_size_summary.csv",
        effect_rows,
        [
            "contrast",
            "left",
            "right",
            "left_mean_entropy",
            "right_mean_entropy",
            "delta_entropy",
            "delta_ci_low",
            "delta_ci_high",
            "cliffs_delta",
            "permutation_p",
            "fdr_q",
        ],
    )
    write_markdown(
        RESULTS_DIR / "effect_size_summary.md",
        "# Effect Size Summary",
        effect_md,
    )

    print("Wrote commons_transfer_summary.*, nonllm_reference_policies.*, auxiliary_metrics_summary.*, and effect_size_summary.*")


if __name__ == "__main__":
    main()
