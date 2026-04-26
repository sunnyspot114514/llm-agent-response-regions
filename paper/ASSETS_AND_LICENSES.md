# Assets, Versions, and Licenses

This file lists the main software/runtime assets and external model assets referenced by the submission. The anonymous zip does **not** redistribute model weights; it only documents the assets required to reproduce or inspect the reported evaluation pipeline.

## 1. Local software/runtime assets

### Core analysis environment

- Python `3.13.2`
  - upstream license: Python Software Foundation License
- `llama_cpp` / `llama-cpp-python` `0.3.16`
  - used for the local `llama.cpp` behavior stack
  - upstream license: MIT
- Ollama CLI `0.21.0`
  - used for backend-matched family-line and matched-backend checks
  - upstream license: MIT
- `numpy==2.4.2`
  - upstream license: BSD-3-Clause
- `pandas==2.3.3`
  - upstream license: BSD-3-Clause
- `pydantic==2.11.7`
  - upstream license: MIT
- `matplotlib==3.10.8`
  - upstream license: Matplotlib license
- `scipy==1.16.3`
  - upstream license: BSD-3-Clause

### GPU mechanism environment

- `torch==2.11.0+cu126`
  - upstream license: BSD-3-Clause
- `transformers==4.49.0`
  - upstream license: Apache-2.0
- `accelerate==1.13.0`
  - upstream license: Apache-2.0

## 2. Local model identifiers used by the paper

From `configs/models.yaml`, the local stack references the following families:

- `phi4-mini`
  - local file: `models/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf`
- `qwen3-8b`
  - local file: `models/Qwen_Qwen3-8B-Q4_K_M.gguf`
- `deepseek-r1`
  - local file: `models/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf`
- `mistral-7b`
  - local Ollama blob path
- `phi3`
  - `ollama://phi3:3.8b`
- `phi4-ollama`
  - `ollama://phi4-mini:latest`
- `qwen3-ollama`
  - `ollama://qwen3:8b`
- `qwen35-9b`
  - `ollama://qwen3.5:9b`
- `deepseek-r1-ollama`
  - `ollama://deepseek-r1:8b`
- `llama3-ollama`
  - `ollama://llama3:8b`
- `gemma3-4b`
  - local file + `ollama://gemma3:4b`
- `gemma4-e4b`
  - local file + `ollama://gemma4:e4b`

These identifiers point to locally installed or locally quantized copies. The paper does not redistribute them.

## 3. Upstream model-license summary

The paper relies on upstream model families whose terms are set by their original providers. Review-time reproduction requires the user to obtain access to those models under the corresponding upstream terms.

- `microsoft/Phi-4-mini-instruct`
  - upstream source: Hugging Face model card
  - upstream license: MIT
- `microsoft/Phi-3-mini-4k-instruct`
  - upstream source: Hugging Face model card
  - upstream license: MIT
- `Qwen/Qwen3-8B` and Qwen3.5 family
  - upstream source: Qwen / Hugging Face
  - upstream license family: Apache-2.0
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
  - upstream source: Hugging Face model card
  - upstream license: MIT
- `google/gemma-3-4b-it`
  - upstream source: Hugging Face / Google
  - upstream license: Gemma usage license
- `Gemma 4 E4B`
  - upstream source: Google / Hugging Face / Ollama
  - upstream license: Apache-2.0
- `meta-llama/Meta-Llama-3-8B-Instruct`
  - upstream source: Hugging Face / Meta
  - upstream license: Meta Llama 3 Community License
- `Mistral 7B`
  - upstream source: Mistral / Hugging Face / Ollama
  - upstream license: Apache-2.0

## 4. Terms and boundaries

- The anonymous zip contains **no** third-party model weights.
- Any rerun that requires local inference also requires the reviewer to satisfy the corresponding upstream model terms.
- Local GGUF quantizations used in the experiments should be interpreted as deployment formats of the upstream family, not as newly licensed assets created by this submission.
- The package documents the assets and versions needed to inspect or rerun the benchmark; it does not sublicense or redistribute the external model families.

## 5. Backend/default settings that affect interpretation

For Ollama-backed runs, the benchmark client passes only:

- `temperature`
- `seed`
- `num_ctx`
- `num_predict`
- stop sequences

If a model Modelfile does not override nucleus or repetition controls, the effective engine defaults are:

- `top_p=0.9`
- `top_k=40`
- `repeat_penalty=1.1`

Model-specific overrides used in the paper include:

- Qwen3: `top_p=0.95`, `top_k=20`, `repeat_penalty=1.0`
- Qwen3.5-9B: `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`
- Gemma-3 / Gemma-4: `top_p=0.95`, `top_k=64`
- DeepSeek-R1: `top_p=0.95`
- Phi-3 / Phi-4-mini / Llama3: engine defaults apart from stop tokens

These settings are included here because they are part of the evaluation assumptions, not because the paper claims backend-invariant behavior.
