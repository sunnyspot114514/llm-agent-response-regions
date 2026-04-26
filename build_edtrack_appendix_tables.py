from __future__ import annotations

import csv
import math
import random
from pathlib import Path


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

    print("Wrote commons_transfer_summary.* and nonllm_reference_policies.*")


if __name__ == "__main__":
    main()
