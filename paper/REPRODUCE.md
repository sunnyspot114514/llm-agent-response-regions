# Reproduce the NeurIPS 2026 E&D Submission

This document gives the exact commands, environment assumptions, and output mapping for the anonymous reproducibility package. It is written for review-time inspection of the evaluative claims, not as a claim that the submission is a standalone benchmark product.

## 1. Verified environment

All commands below are intended to be run from the repository root:

```powershell
cd llm-agent-response-regions
```

Primary analysis environment:

- OS: Windows, PowerShell
- Python: `3.13.2`
- `numpy==2.4.2`
- `pandas==2.3.3`
- `pydantic==2.11.7`
- `matplotlib==3.10.8`
- `scipy==1.16.3`
- `llama_cpp==0.3.16`
- Ollama CLI: `0.21.0`

Optional GPU mechanism environment:

- Python environment: `.venv-mech312`
- `torch==2.11.0+cu126`
- `transformers==4.49.0`
- `accelerate==1.13.0`
- GPU used in the paper: `NVIDIA GeForce RTX 3060 12GB`

## 2. Package structure

The anonymized zip contains:

- runtime code: `agents/`, `runtime/`, `schemas/`, `configs/`
- core entry points:
  - `run_batch_experiment.py`
  - `export_data.py`
  - `build_extension_reports.py`
  - `build_edtrack_repair_addendum.py`
  - `build_edtrack_appendix_tables.py`
  - `build_response_region_map.py`
  - `mechanism_pilot_gpu.py`
- bundled canonical outputs in `results/`
- a reduced reviewer log subset in `logs/`
- supporting paper-side docs in `paper/`

## 3. Two supported workflows

### Workflow A: inspect the exact bundled paper outputs

Use this workflow if you want to verify what numbers and figures the paper actually uses.

Inspect these files directly:

- `results/experiment_summary.csv`
- `results/robustness_bootstrap_ci.csv`
- `results/edtrack_repair_addendum.csv`
- `results/commons_transfer_summary.csv`
- `results/auxiliary_metrics_summary.csv`
- `results/effect_size_summary.csv`
- `results/nonllm_reference_policies.csv`
- `results/response_region_map.pdf`
- `results/response_framework_overview.png`
- `results/response_framework_overview.pdf`
- `results/mechanism_phi4_overview.png`

This is the safest workflow for review, because the bundled `results/` directory already contains the exact exported artifacts used by the paper.

### Workflow B: rerun the analysis pipeline on the bundled sample logs

Use this workflow if you want to inspect the mechanics of the export/report pipeline.

Important: the bundled log subset contains only a small reviewer sample. Re-running `export_data.py` will rebuild a **subset** `results/experiment_summary.csv`, not the full 85-condition corpus reported in the paper.

If you want to preserve the exact bundled exports, back them up first:

```powershell
Copy-Item results results_backup -Recurse
```

Then run:

```powershell
py -3.13 export_data.py
py -3.13 build_extension_reports.py
py -3.13 build_edtrack_repair_addendum.py
py -3.13 build_edtrack_appendix_tables.py
py -3.13 build_response_region_map.py
```

## 4. Exact command reference

### 4.1 Rebuild canonical CSV exports from available logs

```powershell
py -3.13 export_data.py
```

Expected outputs:

- `results/episode_summary.csv`
- `results/agent_level_data.csv`
- `results/experiment_summary.csv`

With the bundled sample logs, this reproduces only the included subset of conditions.

### 4.2 Rebuild extension markdown summaries

```powershell
py -3.13 build_extension_reports.py
```

Representative outputs:

- `results/visibility_ablation_summary.md`
- `results/transcript_gradient_summary.md`
- `results/heldout_family_prediction_summary.md`
- `results/commons_task_validation_summary.md`

These reports depend on what experiments are present in `results/experiment_summary.csv`.

### 4.3 Rebuild robustness and classifier addendum

```powershell
py -3.13 build_edtrack_repair_addendum.py
```

Expected outputs:

- `results/edtrack_repair_addendum.csv`
- `results/edtrack_repair_addendum.md`

These support the appendix backend/decode, parser-failure, and classifier-stability checks.

### 4.4 Rebuild appendix transfer/reference tables

```powershell
py -3.13 build_edtrack_appendix_tables.py
```

Expected outputs:

- `results/commons_transfer_summary.csv`
- `results/commons_transfer_summary.md`
- `results/auxiliary_metrics_summary.csv`
- `results/auxiliary_metrics_summary.md`
- `results/effect_size_summary.csv`
- `results/effect_size_summary.md`
- `results/nonllm_reference_policies.csv`
- `results/nonllm_reference_policies.md`

### 4.5 Rebuild the response-label map

```powershell
py -3.13 build_response_region_map.py
```

Expected outputs:

- `results/response_region_map.png`
- `results/response_region_map.pdf`

### 4.6 Optional GPU mechanism rerun

This is **not** required to inspect the main benchmark claims. It requires the separate GPU environment and local access to the upstream Phi weights.

```powershell
.\.venv-mech312\Scripts\python.exe mechanism_pilot_gpu.py --model-id microsoft/Phi-4-mini-instruct --output-basename mechanism_pilot_gpu_phi4 --max-new-tokens 16 --probe-mode full
```

Representative outputs:

- `results/mechanism_pilot_gpu_phi4.json`
- `results/mechanism_pilot_gpu_phi4_summary.md`

## 5. Output-to-paper mapping

- `results/experiment_summary.csv`
  - source for the main corpus counts in Section 3.2
  - source rows behind Table 1 (`tab:core`)
  - source rows behind Table 2 (`tab:regions`)
  - source rows behind Table 3 (`tab:visibility_norms`)
- `results/commons_transfer_summary.csv`
  - source for Table 4 (`tab:commons-transfer`)
- `results/edtrack_repair_addendum.csv`
  - source for Appendix Tables `tab:temp-robustness`, `tab:parser-summary`, `tab:backend-triad-robustness`, and `tab:classifier-robustness`
- `results/auxiliary_metrics_summary.csv`
  - source for Appendix Table `tab:auxiliary-metrics`
- `results/effect_size_summary.csv`
  - source for Appendix Table `tab:effect-sizes`
- `results/nonllm_reference_policies.csv`
  - source for Appendix Table `tab:nonllm-references`
- `results/response_region_map.pdf`
  - source for Figure 2 (`fig:regions`)
- `results/mechanism_phi4_overview.png`
  - source for Figure 3 (`fig:mech`)
- `results/response_framework_overview.png`
  - source for Figure 1 (`fig:overview`)
- `results/response_framework_overview.pdf`
  - vector source for Figure 1 (`fig:overview`)

## 6. What this package does not do

- It does not redistribute model weights.
- It does not include the full 8,500-episode log corpus.
- It does not claim that all reported numbers can be regenerated from the 36 bundled sample episodes alone.
- It does not require reviewers to execute the benchmark in order to assess the paper's claims.

For the exact boundary between directly reproducible subset results and inspectable bundled full-corpus exports, see `paper/WHAT_IS_FULLY_REPRODUCIBLE.md`.
