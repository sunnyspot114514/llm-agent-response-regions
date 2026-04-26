"""Shared helpers for experiment summaries and robustness statistics."""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from config import load_config
from runtime.episode_io import (
    get_agent_action_sequence,
    get_agent_forbidden_count,
    get_episode_agent_ids,
    load_episode_artifact,
)


ACTION_ORDER = ["cooperate", "defect", "defend", "negotiate", "abstain"]
LOGS_DIR = Path("logs")
_RAW_CONFIG_CACHE: dict | None = None


def calculate_entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = len(actions)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def calculate_action_distribution(actions: list[str]) -> list[float]:
    if not actions:
        return [0.0 for _ in ACTION_ORDER]
    counts = Counter(actions)
    total = len(actions)
    return [counts.get(action, 0) / total for action in ACTION_ORDER]


def calculate_transition_entropy(actions: list[str]) -> float:
    if len(actions) < 2:
        return 0.0
    transitions = list(zip(actions[:-1], actions[1:]))
    counts = Counter(transitions)
    total = len(transitions)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def calculate_action_persistence(actions: list[str]) -> float:
    if len(actions) < 2:
        return 1.0
    repeats = sum(1 for left, right in zip(actions[:-1], actions[1:]) if left == right)
    return repeats / (len(actions) - 1)


def js_divergence(distribution_a: list[float], distribution_b: list[float]) -> float:
    midpoint = [(left + right) / 2 for left, right in zip(distribution_a, distribution_b)]
    return 0.5 * kl_divergence(distribution_a, midpoint) + 0.5 * kl_divergence(distribution_b, midpoint)


def kl_divergence(distribution_a: list[float], distribution_b: list[float]) -> float:
    value = 0.0
    for left, right in zip(distribution_a, distribution_b):
        if left <= 0.0 or right <= 0.0:
            continue
        value += left * math.log2(left / right)
    return value


def list_experiment_names(logs_dir: Path = LOGS_DIR) -> list[str]:
    experiments = []
    for path in sorted(logs_dir.iterdir()) if logs_dir.exists() else []:
        if path.is_dir() and any(path.glob("episode_*.json")):
            experiments.append(path.name)
    return experiments


def get_experiment_config(experiment_name: str) -> dict:
    global _RAW_CONFIG_CACHE
    if _RAW_CONFIG_CACHE is None:
        _RAW_CONFIG_CACHE = load_config()
    return _RAW_CONFIG_CACHE.get("experiments", {}).get(experiment_name, {})


def get_experiment_files(experiment_name: str, logs_dir: Path = LOGS_DIR) -> list[Path]:
    log_dir = logs_dir / experiment_name
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob("episode_*.json"))


def summarize_experiment(experiment_name: str, logs_dir: Path = LOGS_DIR) -> dict | None:
    files = get_experiment_files(experiment_name, logs_dir)
    if not files:
        return None

    agent_entropies: list[float] = []
    forbidden_rates: list[float] = []
    transition_entropies: list[float] = []
    persistence_scores: list[float] = []
    episode_mean_entropies: list[float] = []
    episode_distributions: list[list[float]] = []
    policy_divergences: list[float] = []
    action_counts: Counter[str] = Counter()
    parser_failures = 0

    first_artifact = load_episode_artifact(files[0])
    first_agent_ids = get_episode_agent_ids(first_artifact)
    norm_config = first_artifact.get("norm_config") or {}
    social_exposure_config = first_artifact.get("social_exposure_config") or {}
    observation_config = first_artifact.get("observation_config") or {}
    prompt_config = first_artifact.get("prompt_config") or {}
    task_variant = first_artifact.get("task_variant")
    configured = get_experiment_config(experiment_name)
    configured_forbidden = configured.get("forbidden", [])
    configured_norm_mode = configured.get("norm_mode", "soft" if configured_forbidden else "none")
    configured_social_exposure = configured.get("social_exposure") or {}
    configured_observation = configured.get("observation_config") or {}
    configured_prompt = configured.get("prompt_config") or {}
    configured_task_variant = configured_prompt.get("task_variant", "social_game")

    for ep_file in files:
        artifact = load_episode_artifact(ep_file)
        parser_failures += int(artifact.get("parser_failure_count", 0))

        ep_agent_entropies: list[float] = []
        agent_distributions: list[list[float]] = []
        pooled_actions: list[str] = []

        for agent_id in get_episode_agent_ids(artifact):
            actions = get_agent_action_sequence(artifact, agent_id)
            if not actions:
                continue

            distribution = calculate_action_distribution(actions)
            entropy = calculate_entropy(actions)
            forbidden = get_agent_forbidden_count(artifact, agent_id) / len(actions)

            agent_entropies.append(entropy)
            forbidden_rates.append(forbidden)
            transition_entropies.append(calculate_transition_entropy(actions))
            persistence_scores.append(calculate_action_persistence(actions))
            ep_agent_entropies.append(entropy)
            agent_distributions.append(distribution)
            pooled_actions.extend(actions)
            action_counts.update(actions)

        if ep_agent_entropies:
            episode_mean_entropies.append(sum(ep_agent_entropies) / len(ep_agent_entropies))
        if pooled_actions:
            episode_distributions.append(calculate_action_distribution(pooled_actions))
        if len(agent_distributions) >= 2:
            pairwise = [
                js_divergence(left, right) for left, right in combinations(agent_distributions, 2)
            ]
            if pairwise:
                policy_divergences.append(sum(pairwise) / len(pairwise))

    if not agent_entropies:
        return None

    centroid = [
        sum(distribution[index] for distribution in episode_distributions) / len(episode_distributions)
        for index in range(len(ACTION_ORDER))
    ] if episode_distributions else [0.0 for _ in ACTION_ORDER]
    episode_diversity = (
        sum(js_divergence(distribution, centroid) for distribution in episode_distributions)
        / len(episode_distributions)
        if episode_distributions
        else 0.0
    )

    total_actions = sum(action_counts.values())
    mean_entropy = sum(agent_entropies) / len(agent_entropies)
    std_entropy = math.sqrt(
        sum((value - mean_entropy) ** 2 for value in agent_entropies) / len(agent_entropies)
    )

    return {
        "experiment": experiment_name,
        "num_episodes": len(files),
        "num_agents_per_episode": len(first_agent_ids),
        "forbidden_actions": ",".join(
            norm_config.get("forbidden", configured_forbidden)
        ) or "none",
        "norm_mode": norm_config.get("mode", configured_norm_mode),
        "social_exposure": bool(
            social_exposure_config.get("enabled", configured_social_exposure.get("enabled", False))
        ),
        "task_variant": task_variant or prompt_config.get("task_variant", configured_task_variant),
        "visibility_mode": observation_config.get(
            "visibility_mode", configured_observation.get("visibility_mode", "full")
        ),
        "norm_prompt_variant": prompt_config.get(
            "norm_prompt_variant", configured_prompt.get("norm_prompt_variant", "soft_default")
        ),
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "min_entropy": min(agent_entropies),
        "max_entropy": max(agent_entropies),
        "mean_forbidden_rate": sum(forbidden_rates) / len(forbidden_rates) if forbidden_rates else 0.0,
        "parser_failures": parser_failures,
        "cooperate_rate": action_counts.get("cooperate", 0) / total_actions if total_actions else 0.0,
        "defect_rate": action_counts.get("defect", 0) / total_actions if total_actions else 0.0,
        "defend_rate": action_counts.get("defend", 0) / total_actions if total_actions else 0.0,
        "negotiate_rate": action_counts.get("negotiate", 0) / total_actions if total_actions else 0.0,
        "abstain_rate": action_counts.get("abstain", 0) / total_actions if total_actions else 0.0,
        "mean_transition_entropy": (
            sum(transition_entropies) / len(transition_entropies) if transition_entropies else 0.0
        ),
        "mean_action_persistence": (
            sum(persistence_scores) / len(persistence_scores) if persistence_scores else 0.0
        ),
        "mean_policy_divergence": (
            sum(policy_divergences) / len(policy_divergences) if policy_divergences else 0.0
        ),
        "episode_diversity": episode_diversity,
        "agent_entropies": agent_entropies,
        "episode_mean_entropies": episode_mean_entropies,
    }


def bootstrap_mean_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    mean_value = sum(values) / len(values)
    samples = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return mean_value, low, high


def permutation_test_mean(values_a: list[float], values_b: list[float], n_perm: int = 5000, seed: int = 42) -> float:
    if not values_a or not values_b:
        return 1.0
    rng = random.Random(seed)
    observed = abs((sum(values_a) / len(values_a)) - (sum(values_b) / len(values_b)))
    pooled = list(values_a) + list(values_b)
    cutoff = len(values_a)
    exceed = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_a = pooled[:cutoff]
        perm_b = pooled[cutoff:]
        diff = abs((sum(perm_a) / len(perm_a)) - (sum(perm_b) / len(perm_b)))
        if diff >= observed:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)
