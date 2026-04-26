# LLM Agent Response Regions

This repository contains an anonymous reproducibility package for a NeurIPS 2026 Evaluations & Datasets submission. It supports a controlled empirical evaluation of behavioral support concentration in local LLM interactions. It is not a standalone model release and does not redistribute model weights.

## What Is Included

- Runtime code for constrained local-agent interaction experiments: `agents/`, `runtime/`, `schemas/`, `configs/`.
- Analysis and figure builders:
  - `export_data.py`
  - `build_extension_reports.py`
  - `build_edtrack_repair_addendum.py`
  - `build_edtrack_appendix_tables.py`
  - `build_response_region_map.py`
  - `build_response_framework_figure.py`
- Canonical exported result summaries in `results/`.
- Representative sample logs in `logs/`.
- Reproducibility documentation in `paper/`.

## Quick Start

Create a Python 3.13 environment and install the analysis dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Inspect the bundled paper-side outputs directly:

```powershell
Get-ChildItem results
```

Rebuild the analysis outputs from the bundled sample logs:

```powershell
Copy-Item results results_backup -Recurse
.\.venv\Scripts\python.exe export_data.py
.\.venv\Scripts\python.exe build_extension_reports.py
.\.venv\Scripts\python.exe build_edtrack_repair_addendum.py
.\.venv\Scripts\python.exe build_edtrack_appendix_tables.py
.\.venv\Scripts\python.exe build_response_region_map.py
.\.venv\Scripts\python.exe build_response_framework_figure.py
```

Important: the bundled sample logs are a review-time subset. Re-running `export_data.py` on this repository reconstructs a valid subset export, not the full corpus reported in the manuscript. The exact boundary is documented in `paper/WHAT_IS_FULLY_REPRODUCIBLE.md`.

## Model Access

The repository does not include model checkpoints. Local model identifiers and backend assumptions are documented in `configs/models.yaml`, `paper/ASSETS_AND_LICENSES.md`, and `paper/REPRODUCE.md`. Users should obtain models from their upstream providers and comply with the corresponding licenses.

## Reproducibility Documents

- `paper/REPRODUCE.md`: exact commands, environment assumptions, and output-to-paper mapping.
- `paper/WHAT_IS_FULLY_REPRODUCIBLE.md`: what can be regenerated from the bundled subset versus what is included as canonical exported output.
- `paper/ASSETS_AND_LICENSES.md`: model/backend asset notes and licensing boundaries.
- `paper/COMPUTE.md`: compute estimates for behavior runs, robustness checks, and mechanism follow-up.

## License

Code in this repository is released under the MIT License. Model checkpoints, inference backends, and third-party packages remain governed by their upstream licenses.
