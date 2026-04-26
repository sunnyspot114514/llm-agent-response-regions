# NeurIPS 2026 E&D Reproducibility Manifest

This document describes the anonymized reproducibility package bundled with the submission. It is intended to support the paper's methodological and empirical evaluation claims rather than to claim a submission-time benchmark-suite contribution.

## Suggested bundle structure

- `agents/`
- `runtime/`
- `schemas/`
- `configs/`
- `logs/` or a reduced reviewer subset
- `results/`
- `paper/`

## Package-side documentation

- `paper/REPRODUCE.md`
- `paper/WHAT_IS_FULLY_REPRODUCIBLE.md`
- `paper/ASSETS_AND_LICENSES.md`
- `paper/COMPUTE.md`

## Core analysis entry points

- `run_batch_experiment.py`
  Runs benchmark conditions from `configs/experiment.yaml`.
- `export_data.py`
  Builds the canonical `results/*.csv` exports from logged episodes.
- `build_extension_reports.py`
  Rebuilds markdown summaries for the benchmark extensions.
- `build_edtrack_repair_addendum.py`
  Rebuilds the backend/decode, parser, and classifier addenda.
- `build_edtrack_appendix_tables.py`
  Rebuilds the commons-transfer, auxiliary-metric, effect-size, and non-LLM reference tables used in the appendix.
- `build_response_region_map.py`
  Rebuilds the response-region figure.
- `mechanism_pilot_gpu.py`
  Runs the narrow Phi mechanism follow-up.

## Minimal analysis workflow

1. Inspect the bundled canonical outputs in `results/`.
2. If you want to test the pipeline mechanics on the reduced log slice, run `export_data.py`.
3. Run `build_extension_reports.py`.
4. Run `build_edtrack_repair_addendum.py` and `build_edtrack_appendix_tables.py`.
5. Run `build_response_region_map.py`.
6. For the GPU mechanism probe, use the separate environment documented in `paper/REPRODUCE.md`.

## Expected outputs

- `results/experiment_summary.csv`
- `results/robustness_bootstrap_ci.csv`
- `results/commons_transfer_summary.csv`
- `results/auxiliary_metrics_summary.csv`
- `results/effect_size_summary.csv`
- `results/edtrack_assumption_robustness.md`
- `results/edtrack_repair_addendum.md`
- `results/nonllm_reference_policies.csv`
- `results/response_region_map.pdf`
- `results/mechanism_phi4_overview.png`

## Assumptions to document with the package

- Local serving backend and exact model identifiers
- Temperature / seed settings
- Effective Ollama engine defaults and model-specific Modelfile overrides
- JSON-only constrained decoding
- Episode count and round count
- Whether logs are full or reviewer-reduced
- Which outputs are directly reproducible from bundled sample logs versus only inspectable as bundled canonical exports
