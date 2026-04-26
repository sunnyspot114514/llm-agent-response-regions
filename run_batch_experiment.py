"""Batch runner for configured experiments."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from loguru import logger

from agents.lazy_agent import LazyAgent
from config import load_config
from runtime.batch_progress import get_pending_episode_ids, save_batch_summary
from runtime.game_loop import GameLoop
from runtime.memory_bus import MemoryBus
from runtime.social_exposure import build_scripted_peers, make_social_transcript_provider
from schemas.action import ActionSpace


def setup_cpu_limit() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threads", "-t", type=int, default=None)
    args, _ = parser.parse_known_args()

    n_threads = args.threads or 6
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        affinity_mask = (1 << n_threads) - 1
        handle = kernel32.GetCurrentProcess()
        kernel32.SetProcessAffinityMask(handle, affinity_mask)
    except Exception:
        pass

    return n_threads


_N_THREADS = setup_cpu_limit()


def setup_logging(log_root: Path, experiment_name: str) -> Path:
    log_dir = log_root / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add(log_dir / "batch_run.log", level="DEBUG", rotation="50 MB")
    return log_dir


def create_agents(
    model_config: dict,
    agent_configs: list[dict],
    n_threads: int,
    prompt_config: dict | None = None,
) -> list[LazyAgent]:
    agents: list[LazyAgent] = []
    shareable_paths = [
        model_config[agent_cfg["model"]]["path"]
        for agent_cfg in agent_configs
        if model_config[agent_cfg["model"]].get("backend", "llama_cpp") == "llama_cpp"
    ]
    shared_paths = {
        model_path
        for model_path in shareable_paths
        if shareable_paths.count(model_path) > 1
    }
    for agent_cfg in agent_configs:
        model_key = agent_cfg["model"]
        model_cfg = model_config[model_key]
        model_path = model_cfg["path"]
        backend = model_cfg.get("backend", "llama_cpp")
        temperature = agent_cfg.get("temperature", model_cfg.get("default_temperature", 0.7))
        agents.append(
            LazyAgent(
                agent_id=agent_cfg["id"],
                model_path=model_path,
                n_ctx=model_cfg.get("n_ctx", 2048),
                n_threads=n_threads,
                temperature=temperature,
                seed=42,
                cache_model=True,
                prompt_config=prompt_config,
                share_model_across_agents=backend == "llama_cpp" and model_path in shared_paths,
                backend=backend,
                ollama_model=model_cfg.get("ollama_model"),
            )
        )
    return agents


def create_action_space(config: dict, exp_cfg: dict) -> ActionSpace:
    allowed = config.get("action_space", {}).get(
        "allowed", ["cooperate", "defect", "defend", "negotiate", "abstain"]
    )
    return ActionSpace(
        allowed=allowed,
        forbidden=exp_cfg.get("forbidden", []),
        norm_mode=exp_cfg.get("norm_mode", "soft"),
        norm_rules=exp_cfg.get("norm_rules", []),
    )


def run_batch(
    experiment_name: str,
    num_episodes: int,
    rounds_per_episode: int,
    agents: list[LazyAgent],
    action_space: ActionSpace,
    log_dir: Path,
    social_exposure_config: dict | None = None,
    observation_config: dict | None = None,
):
    pending_episode_ids = get_pending_episode_ids(log_dir, num_episodes)
    if pending_episode_ids:
        logger.info(
            f"Pending episodes: {len(pending_episode_ids)} "
            f"(starting from episode {pending_episode_ids[0]})"
        )
    else:
        logger.info(f"All {num_episodes} episodes already completed, skipping")
        return []

    observer_agents = build_scripted_peers(social_exposure_config)
    social_transcript_provider = make_social_transcript_provider(social_exposure_config)

    results = []
    start_time = time.time()

    for index, ep_id in enumerate(pending_episode_ids, start=1):
        ep_start = time.time()
        seed = 42 + ep_id

        for agent in agents:
            agent.seed = seed
            agent.reset()

        memory_bus = MemoryBus(log_dir=log_dir)
        game = GameLoop(
            agents=agents,
            action_space=action_space,
            max_rounds=rounds_per_episode,
            memory_bus=memory_bus,
            observer_agents=observer_agents,
            social_transcript_provider=social_transcript_provider,
            social_exposure_config=social_exposure_config,
            observation_config=observation_config,
        )
        episode = game.run_episode(ep_id, seed=seed)
        ep_time = time.time() - ep_start

        results.append(
            {
                "episode_id": ep_id,
                "seed": seed,
                "total_rounds": episode.total_rounds,
                "forbidden_triggers": episode.total_forbidden_triggers,
                "parser_failures": episode.parser_failure_count,
                "duration_sec": round(ep_time, 1),
            }
        )

        logger.info(
            f"Episode {ep_id + 1}/{num_episodes} done: "
            f"{episode.total_forbidden_triggers} forbidden, "
            f"{episode.parser_failure_count} parse failures, "
            f"{ep_time:.1f}s"
        )

        if index % 10 == 0:
            save_batch_summary(log_dir, experiment_name, results, num_episodes)

    total_time = time.time() - start_time
    save_batch_summary(log_dir, experiment_name, results, num_episodes)
    logger.info(f"Batch complete: {len(results)} new episodes in {total_time / 60:.1f} min")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", "-t", type=int, default=None, help="CPU threads")
    parser.add_argument("--episodes", "-n", type=int, default=50, help="Episodes per experiment")
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only run the specified experiment names from configs/experiment.yaml",
    )
    args = parser.parse_args()

    config = load_config()
    model_config = config["models"]
    exp_config = config["experiments"]
    log_root = Path(config.get("logs_dir", "logs"))
    n_threads = args.threads or _N_THREADS

    logger.info(f"Using {n_threads} threads")

    if args.only:
        experiments_to_run = [(name, args.episodes) for name in args.only]
    else:
        experiments_to_run = [
            ("baseline", args.episodes),
            ("single_norm", args.episodes),
            ("multi_free", args.episodes),
            ("multi_norm", args.episodes),
        ]

    for exp_name, num_episodes in experiments_to_run:
        if exp_name not in exp_config:
            logger.warning(f"Experiment {exp_name} not found in config, skipping")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Starting {exp_name}: {num_episodes} episodes")
        logger.info(f"{'=' * 60}")

        exp_cfg = exp_config[exp_name]
        log_dir = setup_logging(log_root, exp_name)
        action_space = create_action_space(config, exp_cfg)
        agents = create_agents(
            model_config,
            exp_cfg["agents"],
            n_threads,
            prompt_config=exp_cfg.get("prompt_config"),
        )

        try:
            run_batch(
                experiment_name=exp_name,
                num_episodes=num_episodes,
                rounds_per_episode=exp_cfg.get("rounds", 30),
                agents=agents,
                action_space=action_space,
                log_dir=log_dir,
                social_exposure_config=exp_cfg.get("social_exposure"),
                observation_config=exp_cfg.get("observation_config"),
            )
            logger.info(f"{exp_name} complete!")
        finally:
            for agent in agents:
                agent.close()


if __name__ == "__main__":
    main()
