"""Build markdown summaries for the six framework-extension experiment groups."""

from __future__ import annotations

from pathlib import Path

from analysis_utils import summarize_experiment


RESULTS_DIR = Path("results")


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_visibility_report():
    experiments = ["multi_free", "multi_free_blind", "multi_free_agg", "multi_norm", "multi_norm_blind", "multi_norm_agg"]
    rows = [summarize_experiment(name) for name in experiments]
    rows = [row for row in rows if row is not None]
    if not rows:
        return

    lines = [
        "# Visibility Ablation Summary",
        "",
        "| Condition | Visibility | Episodes | Mean Entropy | Action Persistence | Policy Divergence | Cooperate Rate | Defect Rate |",
        "|-----------|------------|----------|--------------|--------------------|-------------------|----------------|-------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['visibility_mode']} | {row['num_episodes']} | {row['mean_entropy']:.3f} | "
            f"{row['mean_action_persistence']:.3f} | {row['mean_policy_divergence']:.3f} | "
            f"{row['cooperate_rate']:.3f} | {row['defect_rate']:.3f} |"
        )
    write_text(RESULTS_DIR / "visibility_ablation_summary.md", "\n".join(lines) + "\n")


def build_transcript_gradient_report():
    mapping = [
        ("Phi", "solo_phi_exposed", "phi_tx_mixed", "phi_tx_all_defect", "phi_tx_diverse"),
        ("DeepSeek", "solo_ds_exposed", "ds_tx_mixed", "ds_tx_all_defect", "ds_tx_diverse"),
    ]
    lines = [
        "# Transcript Gradient Summary",
        "",
        "| Family | All-Coop | Mixed | All-Defect | Diverse |",
        "|--------|----------|-------|------------|---------|",
    ]
    for family, *experiments in mapping:
        rows = [summarize_experiment(name) for name in experiments]
        if any(row is None for row in rows):
            continue
        entries = [f"`{name}` H={row['mean_entropy']:.3f}" for name, row in zip(experiments, rows)]
        lines.append(f"| {family} | {entries[0]} | {entries[1]} | {entries[2]} | {entries[3]} |")
    write_text(RESULTS_DIR / "transcript_gradient_summary.md", "\n".join(lines) + "\n")


def build_single_norm_prompt_report():
    experiments = [
        "single_norm",
        "single_norm_forbidden_hard",
        "single_norm_forbidden_deontic",
        "single_norm_forbidden_penalty_text",
        "single_norm_positive_reframe",
        "single_norm_forbidden_defend",
        "single_norm_forbidden_abstain",
    ]
    rows = [summarize_experiment(name) for name in experiments]
    rows = [row for row in rows if row is not None]
    if not rows:
        return

    lines = [
        "# Single-Agent Norm Prompt Ablation",
        "",
        "| Condition | Prompt Variant | Forbidden Target | Episodes | Mean Entropy | Forbidden Rate | Cooperate Rate | Defect Rate | Action Persistence |",
        "|-----------|----------------|------------------|----------|--------------|----------------|----------------|-------------|--------------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['norm_prompt_variant']} | {row['forbidden_actions']} | {row['num_episodes']} | "
            f"{row['mean_entropy']:.3f} | {row['mean_forbidden_rate']:.3f} | {row['cooperate_rate']:.3f} | "
            f"{row['defect_rate']:.3f} | {row['mean_action_persistence']:.3f} |"
        )
    write_text(RESULTS_DIR / "single_norm_prompt_ablation_summary.md", "\n".join(lines) + "\n")


def build_heldout_family_report():
    experiments = [
        "solo_mistral",
        "solo_mistral_exposed",
        "homo_mistral",
        "solo_llama3_ollama",
        "solo_llama3_ollama_exposed",
        "homo_llama3_ollama",
        "solo_gemma3_4b",
        "solo_gemma3_4b_exposed",
        "homo_gemma3_4b",
        "solo_gemma4_e4b",
        "solo_gemma4_e4b_exposed",
        "homo_gemma4_e4b",
    ]
    rows = [summarize_experiment(name) for name in experiments]
    rows = [row for row in rows if row is not None]
    if not rows:
        return

    lines = [
        "# Held-Out Family Prediction Summary",
        "",
        "| Condition | Episodes | Mean Entropy | Action Persistence | Cooperate Rate | Defect Rate |",
        "|-----------|----------|--------------|--------------------|----------------|-------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['num_episodes']} | {row['mean_entropy']:.3f} | "
            f"{row['mean_action_persistence']:.3f} | {row['cooperate_rate']:.3f} | {row['defect_rate']:.3f} |"
        )
    write_text(RESULTS_DIR / "heldout_family_prediction_summary.md", "\n".join(lines) + "\n")


def build_gemma_forecast_report():
    groups = [
        ("Gemma 3 4B", "solo_gemma3_4b", "solo_gemma3_4b_exposed", "homo_gemma3_4b"),
        ("Gemma 4 E4B", "solo_gemma4_e4b", "solo_gemma4_e4b_exposed", "homo_gemma4_e4b"),
    ]
    lines = [
        "# Gemma Forecast Summary",
        "",
        "| Family | Solo | Solo Exposed | Homogeneous |",
        "|--------|------|--------------|-------------|",
    ]
    found = False
    for label, solo, exposed, homo in groups:
        rows = [summarize_experiment(name) for name in (solo, exposed, homo)]
        if any(row is None for row in rows):
            continue
        found = True
        lines.append(
            f"| {label} | `{solo}` H={rows[0]['mean_entropy']:.3f} | "
            f"`{exposed}` H={rows[1]['mean_entropy']:.3f} | "
            f"`{homo}` H={rows[2]['mean_entropy']:.3f} |"
        )
    if found:
        write_text(RESULTS_DIR / "gemma_forecast_summary.md", "\n".join(lines) + "\n")


def build_family_line_report():
    groups = [
        ("Qwen3 (Ollama)", "solo_qwen3_ollama", "solo_qwen3_ollama_exposed", "homo_qwen3_ollama"),
        ("Qwen3.5 9B", "solo_qwen35_9b", "solo_qwen35_9b_exposed", "homo_qwen35_9b"),
        ("Phi3 3.8B", "solo_phi3", "solo_phi3_exposed", "homo_phi3"),
        ("Phi4 Mini (Ollama)", "solo_phi4_ollama", "solo_phi4_ollama_exposed", "homo_phi4_ollama"),
    ]
    lines = [
        "# Family Line Validation Summary",
        "",
        "| Family / Version | Solo | Solo Exposed | Homogeneous |",
        "|------------------|------|--------------|-------------|",
    ]
    found = False
    for label, solo, exposed, homo in groups:
        rows = [summarize_experiment(name) for name in (solo, exposed, homo)]
        if any(row is None for row in rows):
            continue
        found = True
        lines.append(
            f"| {label} | `{solo}` H={rows[0]['mean_entropy']:.3f} | "
            f"`{exposed}` H={rows[1]['mean_entropy']:.3f} | "
            f"`{homo}` H={rows[2]['mean_entropy']:.3f} |"
        )
    if found:
        write_text(RESULTS_DIR / "family_line_validation_summary.md", "\n".join(lines) + "\n")


def build_commons_report():
    experiments = [
        "commons_solo_phi",
        "commons_solo_phi_exposed",
        "commons_homo_phi",
        "commons_solo_qwen",
        "commons_solo_qwen_exposed",
        "commons_homo_qwen",
        "commons_solo_ds",
        "commons_solo_ds_exposed",
        "commons_homo_ds",
    ]
    rows = [summarize_experiment(name) for name in experiments]
    rows = [row for row in rows if row is not None]
    if not rows:
        return

    lines = [
        "# Commons Task Validation Summary",
        "",
        "| Condition | Episodes | Mean Entropy | Action Persistence | Policy Divergence | Cooperate Rate | Defect Rate |",
        "|-----------|----------|--------------|--------------------|-------------------|----------------|-------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['num_episodes']} | {row['mean_entropy']:.3f} | "
            f"{row['mean_action_persistence']:.3f} | {row['mean_policy_divergence']:.3f} | "
            f"{row['cooperate_rate']:.3f} | {row['defect_rate']:.3f} |"
        )
    write_text(RESULTS_DIR / "commons_task_validation_summary.md", "\n".join(lines) + "\n")


def build_norm_target_report():
    experiments = [
        "multi_norm",
        "multi_norm_mask",
        "multi_norm_defend",
        "multi_norm_defend_mask",
        "multi_norm_abstain",
        "multi_norm_abstain_mask",
        "multi_norm_bundle_alt",
    ]
    rows = [summarize_experiment(name) for name in experiments]
    rows = [row for row in rows if row is not None]
    if not rows:
        return

    lines = [
        "# Norm Target Generalization Summary",
        "",
        "| Condition | Target | Mode | Episodes | Mean Entropy | Forbidden Rate | Cooperate Rate | Action Persistence |",
        "|-----------|--------|------|----------|--------------|----------------|----------------|--------------------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['forbidden_actions']} | {row['norm_mode']} | {row['num_episodes']} | "
            f"{row['mean_entropy']:.3f} | {row['mean_forbidden_rate']:.3f} | {row['cooperate_rate']:.3f} | "
            f"{row['mean_action_persistence']:.3f} |"
        )
    write_text(RESULTS_DIR / "norm_target_generalization_summary.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    build_visibility_report()
    build_transcript_gradient_report()
    build_single_norm_prompt_report()
    build_heldout_family_report()
    build_gemma_forecast_report()
    build_family_line_report()
    build_commons_report()
    build_norm_target_report()
    print("Extension reports written to results/")
