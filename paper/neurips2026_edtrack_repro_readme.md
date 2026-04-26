# NeurIPS 2026 E&D Anonymous Reproducibility Package

This package supports the evaluative claims in the submission. It is not a standalone benchmark product release.

## Included contents

- Package-side documentation:
  - `paper/REPRODUCE.md`
  - `paper/WHAT_IS_FULLY_REPRODUCIBLE.md`
  - `paper/ASSETS_AND_LICENSES.md`
  - `paper/COMPUTE.md`
- Core runtime code:
  - `agents/`
  - `runtime/`
  - `schemas/`
  - `configs/`
- Main entry points:
  - `run_batch_experiment.py`
  - `export_data.py`
  - `build_extension_reports.py`
  - `build_edtrack_repair_addendum.py`
  - `build_edtrack_appendix_tables.py`
  - `build_response_region_map.py`
  - `mechanism_pilot_gpu.py`
- Exported results:
  - `results/commons_transfer_summary.csv`
  - `results/commons_transfer_summary.md`
  - `results/auxiliary_metrics_summary.csv`
  - `results/auxiliary_metrics_summary.md`
  - `results/effect_size_summary.csv`
  - `results/effect_size_summary.md`
  - `results/experiment_summary.csv`
  - `results/robustness_bootstrap_ci.csv`
  - `results/edtrack_repair_addendum.csv`
  - `results/edtrack_repair_addendum.md`
  - `results/nonllm_reference_policies.csv`
  - `results/nonllm_reference_policies.md`
  - `results/response_region_map.pdf`
  - `results/response_framework_overview.png`
  - `results/response_framework_overview.pdf`
  - `results/mechanism_phi4_overview.png`
- Sample logs:
  - representative subsets from core and robustness conditions

## Supported review workflows

### Workflow A: inspect the exact bundled outputs

Inspect the canonical exported files in `results/` directly. This is the safest way to verify the exact paper-side numbers because the package includes the exported full-corpus summaries used by the manuscript.

### Workflow B: rerun the pipeline on the bundled sample logs

From the repository root:

```powershell
py -3.13 export_data.py
py -3.13 build_extension_reports.py
py -3.13 build_edtrack_repair_addendum.py
py -3.13 build_edtrack_appendix_tables.py
py -3.13 build_response_region_map.py
```

For the GPU mechanism follow-up, use the separate environment documented in `paper/REPRODUCE.md`.

## Notes

- The paper reports a larger corpus than the sample logs included here. The sample log subset is provided to keep the anonymous supplement small while still making the pipeline inspectable.
- Re-running `export_data.py` on the bundled sample logs rebuilds a valid **subset** export, not the full 85-condition corpus. The exact boundary between directly reproducible subset results and inspectable full-corpus exports is documented in `paper/WHAT_IS_FULLY_REPRODUCIBLE.md`.
- Exact local model identifiers and backend assumptions are documented in the paper appendix and manifest.
- For Ollama-backed runs, the benchmark client passes only `temperature`, `seed`, `num_ctx`, `num_predict`, and stop sequences. When a model Modelfile does not override nucleus or repetition parameters, the effective engine defaults are `top_p=0.9`, `top_k=40`, and `repeat_penalty=1.1`. In the models used here, Qwen3 overrides to `top_p=0.95`, `top_k=20`, `repeat_penalty=1.0`; Qwen3.5-9B overrides to `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`; Gemma-3 and Gemma-4 override to `top_p=0.95`, `top_k=64`; DeepSeek-R1 overrides to `top_p=0.95`; and Phi-3, Phi-4-mini, and Llama3 leave those controls at engine defaults apart from stop tokens.
