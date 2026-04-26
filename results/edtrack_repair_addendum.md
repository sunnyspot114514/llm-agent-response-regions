# EDTrack Repair Addendum

This addendum compacts the last-round review fixes into four narrow checks: backend/decode sensitivity, triad-composition robustness, parser-failure concentration, and nearest-anchor classifier stability.

## 1. Backend and decode sensitivity

| Condition | Episodes | Mean H | Persistence | JSD | Parser failures |
|-----------|----------|--------|-------------|-----|-----------------|
| baseline | 100 | 1.421 | 0.201 | 0.000 | 0 |
| single_norm | 100 | 0.046 | 0.997 | 0.000 | 1 |
| single_norm_temp07 | 100 | 0.000 | 1.000 | 0.000 | 0 |
| single_norm_matched | 100 | 0.000 | 1.000 | 0.000 | 0 |
| multi_free | 100 | 0.975 | 0.583 | 0.201 | 0 |
| multi_norm | 100 | 1.006 | 0.509 | 0.272 | 0 |
| temp_low | 100 | 0.005 | 0.999 | 0.658 | 1 |
| temp_mid | 100 | 0.025 | 0.996 | 0.664 | 5 |
| temp_high | 100 | 0.064 | 0.986 | 0.678 | 8 |
| multi_norm_temp07 | 100 | 0.015 | 0.999 | 0.677 | 0 |
| multi_free_matched | 100 | 0.008 | 0.995 | 0.020 | 0 |
| multi_norm_matched | 100 | 0.017 | 0.999 | 0.040 | 0 |
| multi_free_rotated | 100 | 0.028 | 0.981 | 0.011 | 0 |
| multi_norm_rotated | 100 | 0.030 | 0.981 | 0.010 | 0 |
| multi_free_phi3_qwen3_ds_ollama | 100 | 0.058 | 0.973 | 0.055 | 0 |
| multi_norm_phi3_qwen3_ds_ollama | 100 | 0.052 | 0.983 | 0.061 | 0 |
| multi_free_phi4_qwen35_ds_ollama | 100 | 0.008 | 0.995 | 0.020 | 0 |
| multi_norm_phi4_qwen35_ds_ollama | 100 | 0.008 | 1.000 | 0.040 | 0 |
| multi_free_phi4_llama3_ds_ollama | 100 | 0.008 | 0.995 | 0.020 | 0 |
| multi_norm_phi4_llama3_ds_ollama | 100 | 0.008 | 1.000 | 0.040 | 0 |
| solo_qwen | 100 | 0.000 | 1.000 | 0.000 | 0 |
| solo_qwen3_ollama | 100 | 0.000 | 1.000 | 0.000 | 0 |
| solo_phi4_ollama | 100 | 0.000 | 1.000 | 0.000 | 0 |

Reading:

- The local single-agent norm cell is also stack- and temperature-sensitive: `single_norm` at the original 0.5 temperature remains the violating-attractor cell, while `single_norm_temp07` and `single_norm_matched` collapse when available in the current export.
- `multi_free` / `multi_norm` on the original local stack: `H=0.975` / `H=1.006`.
- Matched-temperature local triad checks: `temp_low`, `temp_mid`, `temp_high` at `H=0.005`, `0.025`, `0.064`; `multi_norm_temp07` at `H=0.015`.
- Matched-backend Ollama triad: `multi_free_matched` and `multi_norm_matched` at `H=0.008` and `H=0.017`.
- Order-rotation checks: `multi_free_rotated`, `multi_norm_rotated` at `H=0.028` and `H=0.030`.
- Additional all-Ollama compositions: Phi3/Qwen3/DeepSeek (`H=0.058` / `0.052`), Phi4/Qwen3.5/DeepSeek (`H=0.008` / `0.008`), and Phi4/Llama3/DeepSeek (`H=0.008` / `0.008`).
- The existing family-line controls remain the cleanest within-family backend check: Qwen stays prior-locked on both stacks, while Phi shifts sharply between local `llama.cpp` and Ollama.

## 2. Parser-failure concentration

| Condition | Episodes | Parser failures | Failure rate / action | Mean H |
|-----------|----------|-----------------|-----------------------|--------|
| temp_high | 100 | 8 | 0.0889% | 0.064 |
| temp_mid | 100 | 5 | 0.0556% | 0.025 |
| homo_qwen35_9b | 100 | 4 | 0.0444% | 0.003 |
| solo_qwen35_9b_exposed | 100 | 3 | 0.1000% | 0.006 |
| homo_phi3 | 100 | 1 | 0.0111% | 0.058 |
| multi_free_blind | 100 | 1 | 0.0111% | 0.497 |
| multi_norm_abstain_mask | 100 | 1 | 0.0111% | 0.251 |
| multi_norm_multi | 100 | 1 | 0.0111% | 0.309 |
| multi_norm_strong | 100 | 1 | 0.0111% | 0.616 |
| single_norm | 100 | 1 | 0.0333% | 0.046 |
| solo_gemma4_e4b | 100 | 1 | 0.0333% | 0.012 |
| temp_low | 100 | 1 | 0.0111% | 0.005 |

Reading:

- Parser failures remain concentrated in the temperature pilot, especially `temp_high`, rather than in the main benchmark conditions.
- The new matched-backend robustness runs have zero parser failures, so they isolate backend/condition effects without introducing extra parse noise.
- The temperature pilot should therefore be read as a small diagnostic check rather than as evidence of temperature-invariant response regions.

## 3. Nearest-anchor classifier robustness

| Family | Entropy only | Entropy + persistence | Entropy + homo-JSD |
|--------|--------------|-----------------------|--------------------|
| mistral | prior_locked | prior_locked | prior_locked |
| gemma3 | prior_locked | prior_locked | prior_locked |
| gemma4 | prior_locked | prior_locked | prior_locked |
| qwen3_ollama | prior_locked | prior_locked | prior_locked |
| qwen35_9b | prior_locked | prior_locked | prior_locked |
| phi3 | prior_locked | prior_locked | prior_locked |
| phi4_ollama | prior_locked | prior_locked | prior_locked |
| llama3_ollama | prior_locked | prior_locked | prior_locked |

Feature-set agreement against entropy-only labels:

- `entropy_plus_persistence`: 8/8 label matches.
- `entropy_plus_homo_jsd`: 8/8 label matches.

Alternative prior-anchor sensitivity (swapping the prior-locked anchor among exact-zero prior families):

- `qwen_local`: 0/8 label flips.
- `gemma3`: 0/8 label flips.
- `qwen3_ollama`: 0/8 label flips.
- `llama3`: 0/8 label flips.

Overall reading:

- The nearest-anchor rule remains intentionally simple, but these checks show it is not purely ad hoc: adding persistence or homogeneous JSD does not materially change the held-out labels in the current set.
- Sensitivity on the prior-locked side is especially low because multiple held-out families share the same exact zero corner.
