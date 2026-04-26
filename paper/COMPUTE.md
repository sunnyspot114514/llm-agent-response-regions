# Compute Summary

This file gives the compute posture of the submission in the form expected by the NeurIPS E&D checklist: what hardware was used, what kind of workloads were run, and the approximate order of compute consumed by each class of experiment.

## 1. Hardware posture

The paper was produced on a single local workstation rather than on a cluster.

Behavior corpus:

- local `llama.cpp` and Ollama backends
- CPU-threaded behavior runs, typically with `n_threads=6`
- context window in `configs/models.yaml`: `n_ctx=2048`

Mechanism follow-up:

- one local GPU: `NVIDIA GeForce RTX 3060 12GB`
- CUDA PyTorch environment
- no training or fine-tuning; only forward-pass probing and patching

## 2. Representative measured wall-clock

The project was run incrementally over many batches, so the table below should be read as a practical order-of-magnitude estimate rather than as a token-metered cloud accounting ledger.

Representative batch durations taken from the run logs:

- `solo_phi_exposed`
  - `95` new episodes in `84.5` minutes
  - about `0.89` minutes per episode
- `homo_ds`
  - late-run episodes around `129`--`155` seconds each
  - about `2.1`--`2.6` minutes per episode
- `multi_free` on the older pre-sharing stack
  - `100` episodes in `2287.8` minutes
  - about `22.9` minutes per episode
- `multi_free_matched`
  - `97` new episodes in `911.8` minutes
  - about `9.4` minutes per episode
- `multi_norm_rotated`
  - `97` new episodes in `613.5` minutes
  - about `6.3` minutes per episode

These logs support the practical summary used throughout the project:

- single-agent runs are on the order of `~1 minute/episode`
- shared-model homogeneous runs are on the order of `~2 to 3 minutes/episode`
- matched-backend triad checks are on the order of `~6 to 10 minutes/episode`
- older heterogeneous triad runs on the pre-sharing stack could be much slower

## 3. Corpus-scale compute estimate

The corrected corpus reported in the paper contains:

- `85 conditions`
- `8,500 episodes`
- `16,900 agent trajectories`
- `507,000 logged actions`

Because the corpus mixes single-agent, homogeneous, heterogeneous, matched-backend, and follow-up conditions, there is no single per-episode rate that applies uniformly. A practical summary is:

- the behavior corpus consumed **several hundred single-workstation wall-clock hours**
- a conservative order-of-magnitude estimate for the full project behavior runs is roughly **500--850 wall-clock hours**

This estimate excludes:

- failed exploratory runs that were discarded before inclusion
- paper-writing / figure-editing time
- PDF compilation

## 4. Mechanism follow-up compute

The mechanism section is intentionally narrow compared with the behavior corpus:

- one-family focus (`Phi`)
- one local GPU (`RTX 3060 12GB`)
- no training
- no large hyperparameter sweep

The GPU probes are therefore small compared with the main corpus and should be understood as a modest follow-up cost on top of the much larger CPU-side behavior evaluation.

## 5. What this document is and is not

This is an engineering compute summary for review-time transparency.

It is **not**:

- a cloud billing record
- an energy audit
- a promise that every reviewer must rerun the full corpus

The intended review posture is:

- inspect bundled exported results directly
- inspect a reduced log subset
- rerun small analysis scripts if desired
- treat full-corpus regeneration as a local rerun task requiring the model stack described in `paper/ASSETS_AND_LICENSES.md`
