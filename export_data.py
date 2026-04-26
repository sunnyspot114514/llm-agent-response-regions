"""Export verified episode-, agent-, and experiment-level CSV files."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from analysis_utils import calculate_entropy, list_experiment_names, summarize_experiment
from runtime.episode_io import (
    get_agent_action_sequence,
    get_agent_forbidden_count,
    get_episode_agent_ids,
    get_rounds,
    load_episode_artifact,
)


def export_episode_summary(output_path: Path = Path("results")):
    output_path.mkdir(exist_ok=True)
    rows = []

    for exp_name in list_experiment_names():
        for ep_file in sorted((Path("logs") / exp_name).glob("episode_*.json")):
            artifact = load_episode_artifact(ep_file)
            ep_id = int(ep_file.stem.split("_")[1])

            agent_entropies = {}
            agent_forbidden = {}
            all_actions = []

            for agent_id in get_episode_agent_ids(artifact):
                actions = get_agent_action_sequence(artifact, agent_id)
                if not actions:
                    continue
                forbidden = get_agent_forbidden_count(artifact, agent_id)
                agent_entropies[agent_id] = calculate_entropy(actions)
                agent_forbidden[agent_id] = forbidden
                all_actions.extend(actions)

            if not agent_entropies:
                continue

            rows.append(
                {
                    "experiment": exp_name,
                    "task_variant": artifact.get("task_variant") or "social_game",
                    "visibility_mode": (artifact.get("observation_config") or {}).get(
                        "visibility_mode", "full"
                    ),
                    "episode_id": ep_id,
                    "num_agents": len(agent_entropies),
                    "num_rounds": len(get_rounds(artifact)),
                    "mean_agent_entropy": sum(agent_entropies.values()) / len(agent_entropies),
                    "system_entropy": calculate_entropy(all_actions),
                    "total_forbidden": sum(agent_forbidden.values()),
                }
            )

    csv_path = output_path / "episode_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported {len(rows)} episodes to {csv_path}")
    return rows


def export_agent_level_data(output_path: Path = Path("results")):
    output_path.mkdir(exist_ok=True)
    rows = []

    for exp_name in list_experiment_names():
        for ep_file in sorted((Path("logs") / exp_name).glob("episode_*.json")):
            artifact = load_episode_artifact(ep_file)
            ep_id = int(ep_file.stem.split("_")[1])

            for agent_id in get_episode_agent_ids(artifact):
                actions = get_agent_action_sequence(artifact, agent_id)
                if not actions:
                    continue

                forbidden = get_agent_forbidden_count(artifact, agent_id)
                counter = Counter(actions)
                rows.append(
                    {
                        "experiment": exp_name,
                        "task_variant": artifact.get("task_variant") or "social_game",
                        "visibility_mode": (artifact.get("observation_config") or {}).get(
                            "visibility_mode", "full"
                        ),
                        "episode_id": ep_id,
                        "agent_id": agent_id,
                        "entropy": calculate_entropy(actions),
                        "forbidden_count": forbidden,
                        "forbidden_rate": forbidden / len(actions),
                        "cooperate": counter.get("cooperate", 0),
                        "defect": counter.get("defect", 0),
                        "defend": counter.get("defend", 0),
                        "negotiate": counter.get("negotiate", 0),
                        "abstain": counter.get("abstain", 0),
                    }
                )

    csv_path = output_path / "agent_level_data.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported {len(rows)} agent records to {csv_path}")
    return rows


def export_experiment_summary(output_path: Path = Path("results")):
    output_path.mkdir(exist_ok=True)
    rows = []

    for exp_name in list_experiment_names():
        summary = summarize_experiment(exp_name)
        if summary is None:
            continue
        rows.append(
            {
                "experiment": summary["experiment"],
                "num_agents": summary["num_agents_per_episode"],
                "forbidden_actions": summary["forbidden_actions"],
                "norm_mode": summary["norm_mode"],
                "social_exposure": summary["social_exposure"],
                "task_variant": summary["task_variant"],
                "visibility_mode": summary["visibility_mode"],
                "norm_prompt_variant": summary["norm_prompt_variant"],
                "num_episodes": summary["num_episodes"],
                "mean_entropy": summary["mean_entropy"],
                "std_entropy": summary["std_entropy"],
                "min_entropy": summary["min_entropy"],
                "max_entropy": summary["max_entropy"],
                "mean_forbidden_rate": summary["mean_forbidden_rate"],
                "mean_transition_entropy": summary["mean_transition_entropy"],
                "mean_action_persistence": summary["mean_action_persistence"],
                "mean_policy_divergence": summary["mean_policy_divergence"],
                "episode_diversity": summary["episode_diversity"],
                "parser_failures": summary["parser_failures"],
            }
        )

    csv_path = output_path / "experiment_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Exported {len(rows)} experiment summaries to {csv_path}")
    return rows


if __name__ == "__main__":
    export_episode_summary()
    export_agent_level_data()
    export_experiment_summary()
    print("\nAll data exported to results/ directory")
