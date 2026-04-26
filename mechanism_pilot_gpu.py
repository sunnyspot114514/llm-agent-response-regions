"""GPU-backed hidden-state mechanism probes for Phi-family prompts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F
import transformers.utils as transformers_utils
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from runtime.prompt_builder import build_agent_prompt
from runtime.social_exposure import build_scripted_peers
from schemas.action import Action, ActionSpace
from schemas.agent_state import AgentState
from schemas.world_state import WorldState


ACTIONS = ["cooperate", "defect", "defend", "negotiate", "abstain"]
RESULTS_DIR = Path("results")
DEFAULT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def ensure_phi_transformers_compat():
    if hasattr(transformers_utils, "LossKwargs"):
        return

    class LossKwargs(TypedDict, total=False):
        """Compatibility shim for Phi remote code on older transformers builds."""

    transformers_utils.LossKwargs = LossKwargs


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_phi_prompt(
    *,
    action_space: ActionSpace,
    prompt_config: dict | None = None,
    scripted_peers: list | None = None,
    last_actions: dict[str, Action] | None = None,
    round_id: int = 1,
) -> str:
    return build_agent_prompt(
        agent_id="agent_0",
        action_space=action_space,
        prompt_config=prompt_config or {},
        world_state=WorldState(current_round=round_id),
        other_agents=scripted_peers or [],
        last_round_actions=last_actions,
        agent_state=AgentState(agent_id="agent_0", model_name="phi-mech-hf"),
        observation_context={"visibility_mode": "full"},
    )


def build_transcript_prompts() -> dict[str, str]:
    action_space = ActionSpace(allowed=ACTIONS)
    return {
        "baseline_no_transcript": build_phi_prompt(
            action_space=action_space,
            last_actions=None,
            round_id=1,
        ),
        "all_coop_transcript": build_phi_prompt(
            action_space=action_space,
            scripted_peers=build_scripted_peers(
                {
                    "enabled": True,
                    "peers": [
                        {"id": "peer_alpha", "action": "cooperate"},
                        {"id": "peer_beta", "action": "cooperate"},
                    ],
                }
            ),
            last_actions={
                "peer_alpha": Action(action_type="cooperate"),
                "peer_beta": Action(action_type="cooperate"),
            },
            round_id=1,
        ),
        "all_defect_transcript": build_phi_prompt(
            action_space=action_space,
            scripted_peers=build_scripted_peers(
                {
                    "enabled": True,
                    "peers": [
                        {"id": "peer_alpha", "action": "defect"},
                        {"id": "peer_beta", "action": "defect"},
                    ],
                }
            ),
            last_actions={
                "peer_alpha": Action(action_type="defect"),
                "peer_beta": Action(action_type="defect"),
            },
            round_id=1,
        ),
    }


def build_wording_prompts() -> dict[str, str]:
    action_space = ActionSpace(
        allowed=ACTIONS,
        forbidden=["defect"],
        norm_mode="soft",
    )
    return {
        "soft_default": build_phi_prompt(
            action_space=action_space,
            prompt_config={},
            last_actions=None,
            round_id=0,
        ),
        "positive_reframe": build_phi_prompt(
            action_space=action_space,
            prompt_config={"norm_prompt_variant": "positive_reframe"},
            last_actions=None,
            round_id=0,
        ),
        "forbidden_hard": build_phi_prompt(
            action_space=action_space,
            prompt_config={"norm_prompt_variant": "forbidden_hard"},
            last_actions=None,
            round_id=0,
        ),
    }


class HFMechanismProbe:
    def __init__(self, model_id: str, device: str, *, local_files_only: bool = False):
        self.model_id = model_id
        self.device = device
        self.local_files_only = local_files_only
        ensure_phi_transformers_compat()
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            if rope_scaling.get("rope_type") == "default" and "short_factor" not in rope_scaling:
                config.rope_scaling = None
            elif "type" not in rope_scaling and "rope_type" in rope_scaling:
                config.rope_scaling = {**rope_scaling, "type": rope_scaling["rope_type"]}
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.layers = self.model.model.layers
        self.mlps = [layer.mlp for layer in self.layers]
        self.num_attention_heads = int(
            getattr(
                self.layers[0].self_attn,
                "num_heads",
                getattr(config, "num_attention_heads"),
            )
        )
        self.head_dim = int(
            getattr(
                self.layers[0].self_attn,
                "head_dim",
                config.hidden_size // self.num_attention_heads,
            )
        )

    def encode(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self.device)

    @torch.inference_mode()
    def _forward_prefix(
        self,
        prefix_text: str,
        *,
        output_hidden_states: bool,
        patch_layer_idx: int | None = None,
        patch_vector: torch.Tensor | None = None,
        patch_positions_from_end: tuple[int, ...] = (1,),
        head_patch_layer_idx: int | None = None,
        head_patch_input: torch.Tensor | None = None,
        head_patch_indices: tuple[int, ...] = (),
        head_patch_positions_from_end: tuple[int, ...] = (1,),
        mlp_patch_layer_idx: int | None = None,
        mlp_patch_value: torch.Tensor | None = None,
        mlp_patch_positions_from_end: tuple[int, ...] = (1,),
    ):
        prefix_ids = self.encode(prefix_text)
        handles = []
        if patch_layer_idx is not None and patch_vector is not None:
            target_layer = self.layers[patch_layer_idx]
            patch_value = patch_vector.to(self.device)
            if patch_value.ndim == 1:
                patch_value = patch_value.unsqueeze(0)

            def patch_hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    hidden_states = output[0].clone()
                    for rel_pos in patch_positions_from_end:
                        target_idx = hidden_states.shape[1] - rel_pos
                        source_idx = patch_value.shape[0] - rel_pos
                        if target_idx < 0 or source_idx < 0:
                            continue
                        hidden_states[:, target_idx, :] = patch_value[source_idx, :].to(hidden_states.dtype)
                    return (hidden_states,) + output[1:]
                hidden_states = output.clone()
                for rel_pos in patch_positions_from_end:
                    target_idx = hidden_states.shape[1] - rel_pos
                    source_idx = patch_value.shape[0] - rel_pos
                    if target_idx < 0 or source_idx < 0:
                        continue
                    hidden_states[:, target_idx, :] = patch_value[source_idx, :].to(hidden_states.dtype)
                return hidden_states

            handles.append(target_layer.register_forward_hook(patch_hook))

        if head_patch_layer_idx is not None and head_patch_input is not None and head_patch_indices:
            target_o_proj = self.layers[head_patch_layer_idx].self_attn.o_proj
            patch_input = head_patch_input.to(self.device)

            def head_patch_hook(_module, inputs):
                hidden_states = inputs[0].clone()
                for rel_pos in head_patch_positions_from_end:
                    target_idx = hidden_states.shape[1] - rel_pos
                    source_idx = patch_input.shape[0] - rel_pos
                    if target_idx < 0 or source_idx < 0:
                        continue
                    for head_idx in head_patch_indices:
                        start = head_idx * self.head_dim
                        end = start + self.head_dim
                        hidden_states[:, target_idx, start:end] = patch_input[source_idx, start:end].to(
                            hidden_states.dtype
                        )
                return (hidden_states,)

            handles.append(target_o_proj.register_forward_pre_hook(head_patch_hook))

        if mlp_patch_layer_idx is not None and mlp_patch_value is not None:
            target_mlp = self.mlps[mlp_patch_layer_idx]
            patch_value = mlp_patch_value.to(self.device)
            if patch_value.ndim == 1:
                patch_value = patch_value.unsqueeze(0)

            def mlp_patch_hook(_module, _inputs, output):
                patched = output.clone()
                for rel_pos in mlp_patch_positions_from_end:
                    target_idx = patched.shape[1] - rel_pos
                    source_idx = patch_value.shape[0] - rel_pos
                    if target_idx < 0 or source_idx < 0:
                        continue
                    patched[:, target_idx, :] = patch_value[source_idx, :].to(patched.dtype)
                return patched

            handles.append(target_mlp.register_forward_hook(mlp_patch_hook))

        try:
            outputs = self.model(
                prefix_ids,
                output_hidden_states=output_hidden_states,
                use_cache=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        return prefix_ids, outputs

    def _score_action_sequences_from_state(self, logits, past_key_values) -> dict:
        sequence_logprobs: dict[str, float] = {}
        for action in ACTIONS:
            continuation_ids = self.encode(action + '"')[0]
            running_logits = logits
            running_past = past_key_values
            total_logprob = 0.0
            for token_id in continuation_ids:
                log_probs = F.log_softmax(running_logits[0], dim=-1)
                total_logprob += float(log_probs[token_id].item())
                next_outputs = self.model(
                    token_id.view(1, 1).to(self.device),
                    past_key_values=running_past,
                    use_cache=True,
                )
                running_logits = next_outputs.logits[:, -1, :]
                running_past = next_outputs.past_key_values
            sequence_logprobs[action] = total_logprob

        max_logprob = max(sequence_logprobs.values())
        unnormalized = {
            action: math.exp(value - max_logprob)
            for action, value in sequence_logprobs.items()
        }
        normalizer = sum(unnormalized.values())
        probabilities = {
            action: unnormalized[action] / normalizer
            for action in ACTIONS
        }
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        return {
            "probabilities": probabilities,
            "logprobs": sequence_logprobs,
            "top_action": ranked[0][0],
            "top_probability": ranked[0][1],
            "margin_vs_second": ranked[0][1] - ranked[1][1],
        }

    @torch.inference_mode()
    def next_token_distribution(
        self,
        prefix_text: str,
        *,
        patch_layer_idx: int | None = None,
        patch_vector: torch.Tensor | None = None,
        patch_positions_from_end: tuple[int, ...] = (1,),
    ) -> dict:
        _, outputs = self._forward_prefix(
            prefix_text,
            output_hidden_states=False,
            patch_layer_idx=patch_layer_idx,
            patch_vector=patch_vector,
            patch_positions_from_end=patch_positions_from_end,
        )
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits[0], dim=-1).detach().float().cpu()
        topk = torch.topk(probs, k=5)
        return {
            "probs": probs,
            "top_tokens": [
                {
                    "token_id": int(idx.item()),
                    "token_text": self.tokenizer.decode([int(idx.item())]),
                    "prob": float(value.item()),
                }
                for value, idx in zip(topk.values, topk.indices)
            ],
        }

    @torch.inference_mode()
    def extract_layer_vectors(self, text: str) -> list[torch.Tensor]:
        input_ids = self.encode(text)
        outputs = self.model(input_ids, output_hidden_states=True, use_cache=False)
        return [hidden[0, -1, :].detach().float().cpu() for hidden in outputs.hidden_states]

    @torch.inference_mode()
    def score_actions(self, prompt: str) -> dict:
        prefix_ids, outputs = self._forward_prefix(
            prompt + '{"action":"',
            output_hidden_states=True,
        )
        scored = self._score_action_sequences_from_state(
            outputs.logits[:, -1, :],
            outputs.past_key_values,
        )
        scored.update(
            {
            "layer_vectors": [
                hidden[0, -1, :].detach().float().cpu()
                    for hidden in outputs.hidden_states
            ],
            "prefix_token_count": int(prefix_ids.shape[1]),
        }
        )
        return scored

    @torch.inference_mode()
    def greedy_trace(self, prompt: str, max_new_tokens: int) -> dict:
        generated = self.encode(prompt + '{"action":"')
        trace = []
        for _ in range(max_new_tokens):
            outputs = self.model(generated, use_cache=False)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits[0], dim=-1)
            topk = torch.topk(log_probs, k=5)
            next_token_id = int(topk.indices[0].item())
            next_token_text = self.tokenizer.decode([next_token_id])
            trace.append(
                {
                    "next_token_text": next_token_text,
                    "top_tokens": [
                        {
                            "token_text": self.tokenizer.decode([int(idx.item())]),
                            "logprob": float(value.item()),
                        }
                        for value, idx in zip(topk.values, topk.indices)
                    ],
                }
            )
            generated = torch.cat(
                [generated, torch.tensor([[next_token_id]], device=self.device)],
                dim=1,
            )
            if next_token_id == self.eos_token_id:
                break
            if next_token_text == '"' and len(trace) >= 3:
                break
        suffix = self.tokenizer.decode(generated[0, -len(trace) :].tolist()) if trace else ""
        return {"decoded_suffix": suffix, "steps": trace}

    @torch.inference_mode()
    def forced_continuation_trace(self, prompt: str, action: str, max_new_tokens: int) -> dict:
        generated = self.encode(prompt + f'{{"action":"{action}","reason":"')
        trace = []
        for _ in range(max_new_tokens):
            outputs = self.model(generated, use_cache=False)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits[0], dim=-1)
            topk = torch.topk(log_probs, k=5)
            next_token_id = int(topk.indices[0].item())
            next_token_text = self.tokenizer.decode([next_token_id])
            trace.append(
                {
                    "next_token_text": next_token_text,
                    "top_tokens": [
                        {
                            "token_text": self.tokenizer.decode([int(idx.item())]),
                            "logprob": float(value.item()),
                        }
                        for value, idx in zip(topk.values, topk.indices)
                    ],
                }
            )
            generated = torch.cat(
                [generated, torch.tensor([[next_token_id]], device=self.device)],
                dim=1,
            )
            if next_token_id == self.eos_token_id:
                break
            if next_token_text == '"' and len(trace) >= 3:
                break
        suffix = self.tokenizer.decode(generated[0, -len(trace) :].tolist()) if trace else ""
        return {"decoded_suffix": suffix, "steps": trace}

    @torch.inference_mode()
    def score_actions_with_layer_patch(
        self,
        *,
        target_prompt: str,
        layer_idx: int,
        patch_vector: torch.Tensor,
        patch_positions_from_end: tuple[int, ...] = (1,),
    ) -> dict:
        _, patched_outputs = self._forward_prefix(
            target_prompt + '{"action":"',
            output_hidden_states=False,
            patch_layer_idx=layer_idx,
            patch_vector=patch_vector,
            patch_positions_from_end=patch_positions_from_end,
        )
        return self._score_action_sequences_from_state(
            patched_outputs.logits[:, -1, :],
            patched_outputs.past_key_values,
        )

    @torch.inference_mode()
    def get_action_prefix_hidden_states(self, prompt: str) -> list[torch.Tensor]:
        _, outputs = self._forward_prefix(
            prompt + '{"action":"',
            output_hidden_states=True,
        )
        return [hidden.detach() for hidden in outputs.hidden_states]

    @torch.inference_mode()
    def score_actions_with_source_prompt_patch(
        self,
        *,
        target_prompt: str,
        source_prompt: str,
        layer_idx: int,
        patch_positions_from_end: tuple[int, ...] = (1,),
    ) -> dict:
        source_hidden_states = self.get_action_prefix_hidden_states(source_prompt)
        return self.score_actions_with_layer_patch(
            target_prompt=target_prompt,
            layer_idx=layer_idx,
            patch_vector=source_hidden_states[layer_idx + 1][0, -1, :],
            patch_positions_from_end=patch_positions_from_end,
        )

    @torch.inference_mode()
    def get_action_prefix_attention_inputs(self, prompt: str, layer_idx: int) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def capture_hook(_module, inputs):
            captured["value"] = inputs[0].detach()

        handle = self.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(capture_hook)
        try:
            self._forward_prefix(
                prompt + '{"action":"',
                output_hidden_states=False,
            )
        finally:
            handle.remove()

        if "value" not in captured:
            raise RuntimeError(f"Failed to capture o_proj input for layer {layer_idx}")
        return captured["value"][0]

    @torch.inference_mode()
    def get_action_prefix_mlp_outputs(self, prompt: str, layer_idx: int) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}

        def capture_hook(_module, _inputs, output):
            captured["value"] = output.detach()

        handle = self.mlps[layer_idx].register_forward_hook(capture_hook)
        try:
            self._forward_prefix(
                prompt + '{"action":"',
                output_hidden_states=False,
            )
        finally:
            handle.remove()

        if "value" not in captured:
            raise RuntimeError(f"Failed to capture mlp output for layer {layer_idx}")
        return captured["value"][0]

    @torch.inference_mode()
    def score_actions_with_head_patch(
        self,
        *,
        target_prompt: str,
        layer_idx: int,
        source_attention_input: torch.Tensor,
        head_indices: tuple[int, ...],
        patch_positions_from_end: tuple[int, ...] = (1,),
    ) -> dict:
        _, patched_outputs = self._forward_prefix(
            target_prompt + '{"action":"',
            output_hidden_states=False,
            head_patch_layer_idx=layer_idx,
            head_patch_input=source_attention_input,
            head_patch_indices=head_indices,
            head_patch_positions_from_end=patch_positions_from_end,
        )
        return self._score_action_sequences_from_state(
            patched_outputs.logits[:, -1, :],
            patched_outputs.past_key_values,
        )

    @torch.inference_mode()
    def score_actions_with_head_set_patch(
        self,
        *,
        target_prompt: str,
        source_attention_inputs: dict[int, torch.Tensor],
        layer_to_head_indices: dict[int, tuple[int, ...]],
    ) -> dict:
        current_prompt = target_prompt + '{"action":"'
        handles = []
        try:
            for layer_idx, head_indices in layer_to_head_indices.items():
                source_attention_input = source_attention_inputs[layer_idx].to(self.device)
                target_o_proj = self.layers[layer_idx].self_attn.o_proj

                def head_patch_hook(
                    _module,
                    inputs,
                    *,
                    _source_attention_input=source_attention_input,
                    _head_indices=head_indices,
                ):
                    hidden_states = inputs[0].clone()
                    target_idx = hidden_states.shape[1] - 1
                    source_idx = _source_attention_input.shape[0] - 1
                    for head_idx in _head_indices:
                        start = head_idx * self.head_dim
                        end = start + self.head_dim
                        hidden_states[:, target_idx, start:end] = _source_attention_input[source_idx, start:end].to(
                            hidden_states.dtype
                        )
                    return (hidden_states,)

                handles.append(target_o_proj.register_forward_pre_hook(head_patch_hook))

            _, patched_outputs = self._forward_prefix(
                current_prompt,
                output_hidden_states=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        return self._score_action_sequences_from_state(
            patched_outputs.logits[:, -1, :],
            patched_outputs.past_key_values,
        )

    @torch.inference_mode()
    def score_actions_with_mlp_patch(
        self,
        *,
        target_prompt: str,
        layer_idx: int,
        mlp_patch_value: torch.Tensor,
        patch_positions_from_end: tuple[int, ...] = (1,),
    ) -> dict:
        _, patched_outputs = self._forward_prefix(
            target_prompt + '{"action":"',
            output_hidden_states=False,
            mlp_patch_layer_idx=layer_idx,
            mlp_patch_value=mlp_patch_value,
            mlp_patch_positions_from_end=patch_positions_from_end,
        )
        return self._score_action_sequences_from_state(
            patched_outputs.logits[:, -1, :],
            patched_outputs.past_key_values,
        )


def summarize_layer_cosines(layer_vectors_by_label: dict[str, list[torch.Tensor]]) -> dict[str, dict]:
    labels = list(layer_vectors_by_label.keys())
    summary = {}
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            values = []
            left_layers = layer_vectors_by_label[left]
            right_layers = layer_vectors_by_label[right]
            for layer_idx in range(len(left_layers)):
                cosine = F.cosine_similarity(
                    left_layers[layer_idx].unsqueeze(0),
                    right_layers[layer_idx].unsqueeze(0),
                ).item()
                values.append(float(cosine))
            mid_idx = len(values) // 2
            min_idx = min(range(len(values)), key=lambda idx: values[idx])
            summary[f"{left}__vs__{right}"] = {
                "layer_0": values[0],
                "layer_mid": values[mid_idx],
                "layer_last": values[-1],
                "min_layer": min_idx,
                "min_cosine": values[min_idx],
            }
    return summary


def probability_l1_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return float(sum(abs(left[action] - right[action]) for action in ACTIONS))


def action_delta_vector(target: dict[str, float], baseline: dict[str, float]) -> torch.Tensor:
    return torch.tensor([float(target[action] - baseline[action]) for action in ACTIONS], dtype=torch.float32)


def summarize_action_shift(
    patched_probs: dict[str, float],
    baseline_probs: dict[str, float],
    source_probs: dict[str, float],
) -> dict[str, object]:
    patched_delta = action_delta_vector(patched_probs, baseline_probs)
    source_delta = action_delta_vector(source_probs, baseline_probs)

    patched_norm = float(torch.norm(patched_delta).item())
    source_norm = float(torch.norm(source_delta).item())
    if patched_norm > 0 and source_norm > 0:
        direction_cosine = float(F.cosine_similarity(patched_delta.unsqueeze(0), source_delta.unsqueeze(0)).item())
    else:
        direction_cosine = 0.0

    deltas = {action: float(patched_delta[idx].item()) for idx, action in enumerate(ACTIONS)}
    largest_positive_action = max(deltas, key=deltas.get)
    largest_negative_action = min(deltas, key=deltas.get)
    return {
        "delta_by_action": deltas,
        "direction_cosine_to_source": direction_cosine,
        "largest_positive_action": largest_positive_action,
        "largest_positive_delta": deltas[largest_positive_action],
        "largest_negative_action": largest_negative_action,
        "largest_negative_delta": deltas[largest_negative_action],
    }


def build_last_n_positions(span_size: int) -> tuple[int, ...]:
    if span_size < 1:
        raise ValueError("span_size must be at least 1")
    return tuple(range(1, span_size + 1))


def select_candidate_layers(best_rows: list[dict], max_layers: int = 4) -> list[int]:
    selected = []
    for row in best_rows:
        layer_idx = int(row["layer"])
        if layer_idx in selected:
            continue
        selected.append(layer_idx)
        if len(selected) >= max_layers:
            break
    return selected


def select_head_sets(best_rows: list[dict], set_sizes: tuple[int, ...] = (2, 4, 8)) -> list[dict]:
    ranked_rows = sorted(best_rows, key=lambda row: row.get("distance_reduction", 0.0), reverse=True)
    ordered_pairs: list[tuple[int, int]] = []
    for row in ranked_rows:
        pair = (int(row["layer"]), int(row["head"]))
        if pair in ordered_pairs:
            continue
        ordered_pairs.append(pair)

    head_sets: list[dict] = []
    for set_size in set_sizes:
        subset = ordered_pairs[:set_size]
        if len(subset) < set_size:
            continue
        by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in subset:
            by_layer.setdefault(layer_idx, []).append(head_idx)
        head_sets.append(
            {
                "set_size": set_size,
                "set_label": f"top_{set_size}_heads",
                "layer_to_heads": {layer: tuple(heads) for layer, heads in by_layer.items()},
                "members": subset,
            }
        )
    return head_sets


def compute_head_overlap_summary(
    left_rows: list[dict],
    right_rows: list[dict],
    k_values: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> dict:
    left_ranked = sorted(left_rows, key=lambda row: row.get("distance_reduction", 0.0), reverse=True)
    right_ranked = sorted(right_rows, key=lambda row: row.get("distance_reduction", 0.0), reverse=True)
    left_ordered: list[tuple[int, int]] = []
    for row in left_ranked:
        pair = (int(row["layer"]), int(row["head"]))
        if pair not in left_ordered:
            left_ordered.append(pair)

    right_ordered: list[tuple[int, int]] = []
    for row in right_ranked:
        pair = (int(row["layer"]), int(row["head"]))
        if pair not in right_ordered:
            right_ordered.append(pair)

    overlaps = []
    for k in k_values:
        left_top = left_ordered[:k]
        right_top = right_ordered[:k]
        if len(left_top) < k or len(right_top) < k:
            continue
        left_set = set(left_top)
        right_set = set(right_top)
        intersection = sorted(left_set & right_set)
        union = left_set | right_set
        overlaps.append(
            {
                "k": k,
                "left_count": len(left_set),
                "right_count": len(right_set),
                "intersection_count": len(intersection),
                "jaccard": len(intersection) / len(union) if union else 0.0,
                "overlap_members": intersection,
            }
        )
    return {
        "k_values": list(k_values),
        "overlaps": overlaps,
    }


def build_activation_patch_summary(
    probe: HFMechanismProbe,
    *,
    target_label: str,
    target_prompt: str,
    source_label: str,
    source_prompt: str,
) -> dict:
    target_baseline = probe.score_actions(target_prompt)
    source_baseline = probe.score_actions(source_prompt)
    source_hidden_states = probe.get_action_prefix_hidden_states(source_prompt)
    source_top_action = source_baseline["top_action"]
    baseline_distance = probability_l1_distance(
        target_baseline["probabilities"],
        source_baseline["probabilities"],
    )

    per_layer = []
    for layer_idx in range(len(probe.layers)):
        patched = probe.score_actions_with_layer_patch(
            target_prompt=target_prompt,
            layer_idx=layer_idx,
            patch_vector=source_hidden_states[layer_idx + 1][0, -1, :],
        )
        patched_distance = probability_l1_distance(
            patched["probabilities"],
            source_baseline["probabilities"],
        )
        per_layer.append(
            {
                "layer": layer_idx,
                "patched_top_action": patched["top_action"],
                "source_top_action": source_top_action,
                "cooperate": patched["probabilities"]["cooperate"],
                "defect": patched["probabilities"]["defect"],
                "negotiate": patched["probabilities"]["negotiate"],
                "source_top_action_delta": patched["probabilities"][source_top_action]
                - target_baseline["probabilities"][source_top_action],
                "distance_reduction": baseline_distance - patched_distance,
            }
        )

    ranked = sorted(
        per_layer,
        key=lambda row: row["distance_reduction"],
        reverse=True,
    )
    return {
        "target_label": target_label,
        "source_label": source_label,
        "target_baseline": {
            "top_action": target_baseline["top_action"],
            "probabilities": target_baseline["probabilities"],
        },
        "source_baseline": {
            "top_action": source_baseline["top_action"],
            "probabilities": source_baseline["probabilities"],
        },
        "baseline_distance_to_source": baseline_distance,
        "best_layers": ranked[:8],
        "all_layers": per_layer,
    }


def build_multi_position_patch_summary(
    probe: HFMechanismProbe,
    *,
    target_label: str,
    target_prompt: str,
    source_label: str,
    source_prompt: str,
    span_sizes: tuple[int, ...] = (1, 2, 4),
) -> dict:
    target_baseline = probe.score_actions(target_prompt)
    source_baseline = probe.score_actions(source_prompt)
    source_hidden_states = probe.get_action_prefix_hidden_states(source_prompt)
    source_top_action = source_baseline["top_action"]
    baseline_distance = probability_l1_distance(
        target_baseline["probabilities"],
        source_baseline["probabilities"],
    )

    per_combo = []
    for span_size in span_sizes:
        positions_from_end = build_last_n_positions(span_size)
        for layer_idx in range(len(probe.layers)):
            patched = probe.score_actions_with_layer_patch(
                target_prompt=target_prompt,
                layer_idx=layer_idx,
                patch_vector=source_hidden_states[layer_idx + 1][0],
                patch_positions_from_end=positions_from_end,
            )
            patched_distance = probability_l1_distance(
                patched["probabilities"],
                source_baseline["probabilities"],
            )
            per_combo.append(
                {
                    "span_size": span_size,
                    "span_label": f"last_{span_size}_tokens",
                    "layer": layer_idx,
                    "patched_top_action": patched["top_action"],
                    "cooperate": patched["probabilities"]["cooperate"],
                    "defect": patched["probabilities"]["defect"],
                    "negotiate": patched["probabilities"]["negotiate"],
                    "source_top_action_delta": patched["probabilities"][source_top_action]
                    - target_baseline["probabilities"][source_top_action],
                    "distance_reduction": baseline_distance - patched_distance,
                }
            )

    ranked = sorted(per_combo, key=lambda row: row["distance_reduction"], reverse=True)
    return {
        "target_label": target_label,
        "source_label": source_label,
        "target_baseline": {
            "top_action": target_baseline["top_action"],
            "probabilities": target_baseline["probabilities"],
        },
        "source_baseline": {
            "top_action": source_baseline["top_action"],
            "probabilities": source_baseline["probabilities"],
        },
        "baseline_distance_to_source": baseline_distance,
        "span_sizes": list(span_sizes),
        "best_combinations": ranked[:12],
        "all_combinations": per_combo,
    }


def build_attention_head_patch_summary(
    probe: HFMechanismProbe,
    *,
    target_label: str,
    target_prompt: str,
    source_label: str,
    source_prompt: str,
    candidate_layers: list[int],
) -> dict:
    target_baseline = probe.score_actions(target_prompt)
    source_baseline = probe.score_actions(source_prompt)
    source_top_action = source_baseline["top_action"]
    baseline_distance = probability_l1_distance(
        target_baseline["probabilities"],
        source_baseline["probabilities"],
    )

    source_attention_inputs = {
        layer_idx: probe.get_action_prefix_attention_inputs(source_prompt, layer_idx)
        for layer_idx in candidate_layers
    }

    per_head = []
    for layer_idx in candidate_layers:
        source_attention_input = source_attention_inputs[layer_idx]
        for head_idx in range(probe.num_attention_heads):
            patched = probe.score_actions_with_head_patch(
                target_prompt=target_prompt,
                layer_idx=layer_idx,
                source_attention_input=source_attention_input,
                head_indices=(head_idx,),
            )
            patched_distance = probability_l1_distance(
                patched["probabilities"],
                source_baseline["probabilities"],
            )
            per_head.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "patched_top_action": patched["top_action"],
                    "cooperate": patched["probabilities"]["cooperate"],
                    "defect": patched["probabilities"]["defect"],
                    "negotiate": patched["probabilities"]["negotiate"],
                    "source_top_action_delta": patched["probabilities"][source_top_action]
                    - target_baseline["probabilities"][source_top_action],
                    "distance_reduction": baseline_distance - patched_distance,
                }
            )

    ranked = sorted(per_head, key=lambda row: row["distance_reduction"], reverse=True)
    return {
        "target_label": target_label,
        "source_label": source_label,
        "target_baseline": {
            "top_action": target_baseline["top_action"],
            "probabilities": target_baseline["probabilities"],
        },
        "source_baseline": {
            "top_action": source_baseline["top_action"],
            "probabilities": source_baseline["probabilities"],
        },
        "baseline_distance_to_source": baseline_distance,
        "candidate_layers": candidate_layers,
        "num_heads": probe.num_attention_heads,
        "head_dim": probe.head_dim,
        "best_heads": ranked[:12],
        "all_heads": per_head,
    }


def build_attention_head_set_patch_summary(
    probe: HFMechanismProbe,
    *,
    target_label: str,
    target_prompt: str,
    source_label: str,
    source_prompt: str,
    base_head_summary: dict,
) -> dict:
    target_baseline = base_head_summary["target_baseline"]
    source_baseline = base_head_summary["source_baseline"]
    baseline_distance = float(base_head_summary["baseline_distance_to_source"])
    source_top_action = source_baseline["top_action"]
    head_sets = select_head_sets(base_head_summary["all_heads"], set_sizes=(1, 2, 4, 8, 16))

    needed_layers = sorted(
        {
            int(layer_idx)
            for head_set in head_sets
            for layer_idx in head_set["layer_to_heads"].keys()
        }
    )
    source_attention_inputs = {
        layer_idx: probe.get_action_prefix_attention_inputs(source_prompt, layer_idx)
        for layer_idx in needed_layers
    }

    results = []
    for head_set in head_sets:
        patched = probe.score_actions_with_head_set_patch(
            target_prompt=target_prompt,
            source_attention_inputs=source_attention_inputs,
            layer_to_head_indices=head_set["layer_to_heads"],
        )
        patched_distance = probability_l1_distance(
            patched["probabilities"],
            source_baseline["probabilities"],
        )
        results.append(
            {
                "set_size": head_set["set_size"],
                "set_label": head_set["set_label"],
                "members": head_set["members"],
                "patched_top_action": patched["top_action"],
                "cooperate": patched["probabilities"]["cooperate"],
                "defect": patched["probabilities"]["defect"],
                "negotiate": patched["probabilities"]["negotiate"],
                "source_top_action_delta": patched["probabilities"][source_top_action]
                - target_baseline["probabilities"][source_top_action],
                "distance_reduction": baseline_distance - patched_distance,
            }
        )

    ranked = sorted(results, key=lambda row: row["distance_reduction"], reverse=True)
    return {
        "target_label": target_label,
        "source_label": source_label,
        "target_baseline": target_baseline,
        "source_baseline": source_baseline,
        "baseline_distance_to_source": baseline_distance,
        "head_sets": head_sets,
        "best_sets": ranked,
        "all_sets": results,
    }


def build_mlp_patch_summary(
    probe: HFMechanismProbe,
    *,
    target_label: str,
    target_prompt: str,
    source_label: str,
    source_prompt: str,
    candidate_layers: list[int],
) -> dict:
    target_baseline = probe.score_actions(target_prompt)
    source_baseline = probe.score_actions(source_prompt)
    source_top_action = source_baseline["top_action"]
    baseline_distance = probability_l1_distance(
        target_baseline["probabilities"],
        source_baseline["probabilities"],
    )

    source_mlp_outputs = {
        layer_idx: probe.get_action_prefix_mlp_outputs(source_prompt, layer_idx)
        for layer_idx in candidate_layers
    }

    per_layer = []
    for layer_idx in candidate_layers:
        patched = probe.score_actions_with_mlp_patch(
            target_prompt=target_prompt,
            layer_idx=layer_idx,
            mlp_patch_value=source_mlp_outputs[layer_idx],
        )
        patched_distance = probability_l1_distance(
            patched["probabilities"],
            source_baseline["probabilities"],
        )
        per_layer.append(
            {
                "layer": layer_idx,
                "patched_top_action": patched["top_action"],
                "cooperate": patched["probabilities"]["cooperate"],
                "defect": patched["probabilities"]["defect"],
                "negotiate": patched["probabilities"]["negotiate"],
                "source_top_action_delta": patched["probabilities"][source_top_action]
                - target_baseline["probabilities"][source_top_action],
                "distance_reduction": baseline_distance - patched_distance,
            }
        )

    ranked = sorted(per_layer, key=lambda row: row["distance_reduction"], reverse=True)
    return {
        "target_label": target_label,
        "source_label": source_label,
        "target_baseline": {
            "top_action": target_baseline["top_action"],
            "probabilities": target_baseline["probabilities"],
        },
        "source_baseline": {
            "top_action": source_baseline["top_action"],
            "probabilities": source_baseline["probabilities"],
        },
        "baseline_distance_to_source": baseline_distance,
        "candidate_layers": candidate_layers,
        "best_layers": ranked,
        "all_layers": per_layer,
    }


def build_semantic_direction_summary(
    probe: HFMechanismProbe,
    *,
    target_label: str,
    target_prompt: str,
    source_label: str,
    source_prompt: str,
    activation_summary: dict,
    head_set_summary: dict,
    mlp_summary: dict,
) -> dict:
    target_baseline = activation_summary["target_baseline"]
    source_baseline = activation_summary["source_baseline"]

    best_residual = activation_summary["best_layers"][0]
    source_hidden_states = probe.get_action_prefix_hidden_states(source_prompt)
    residual_patched = probe.score_actions_with_layer_patch(
        target_prompt=target_prompt,
        layer_idx=int(best_residual["layer"]),
        patch_vector=source_hidden_states[int(best_residual["layer"]) + 1][0, -1, :],
    )

    top_8_set = next(row for row in head_set_summary["best_sets"] if int(row["set_size"]) == 8)
    top_8_head_set = next(
        head_set for head_set in head_set_summary["head_sets"] if int(head_set["set_size"]) == 8
    )
    source_attention_inputs = {
        int(layer_idx): probe.get_action_prefix_attention_inputs(source_prompt, int(layer_idx))
        for layer_idx in top_8_head_set["layer_to_heads"].keys()
    }
    head_set_patched = probe.score_actions_with_head_set_patch(
        target_prompt=target_prompt,
        source_attention_inputs=source_attention_inputs,
        layer_to_head_indices=top_8_head_set["layer_to_heads"],
    )

    best_mlp = mlp_summary["best_layers"][0]
    source_mlp_value = probe.get_action_prefix_mlp_outputs(source_prompt, int(best_mlp["layer"]))
    mlp_patched = probe.score_actions_with_mlp_patch(
        target_prompt=target_prompt,
        layer_idx=int(best_mlp["layer"]),
        mlp_patch_value=source_mlp_value,
    )

    rows = []
    for component_label, patched, distance_reduction in [
        (f"best_residual_L{best_residual['layer']}", residual_patched, float(best_residual["distance_reduction"])),
        ("top_8_heads", head_set_patched, float(top_8_set["distance_reduction"])),
        (f"best_mlp_L{best_mlp['layer']}", mlp_patched, float(best_mlp["distance_reduction"])),
    ]:
        semantic = summarize_action_shift(
            patched["probabilities"],
            target_baseline["probabilities"],
            source_baseline["probabilities"],
        )
        rows.append(
            {
                "component": component_label,
                "distance_reduction": distance_reduction,
                "direction_cosine_to_source": semantic["direction_cosine_to_source"],
                "largest_positive_action": semantic["largest_positive_action"],
                "largest_positive_delta": semantic["largest_positive_delta"],
                "largest_negative_action": semantic["largest_negative_action"],
                "largest_negative_delta": semantic["largest_negative_delta"],
                **{f"delta_{action}": semantic["delta_by_action"][action] for action in ACTIONS},
            }
        )

    source_semantic = summarize_action_shift(
        source_baseline["probabilities"],
        target_baseline["probabilities"],
        source_baseline["probabilities"],
    )
    return {
        "target_label": target_label,
        "source_label": source_label,
        "source_shift": {
            "largest_positive_action": source_semantic["largest_positive_action"],
            "largest_positive_delta": source_semantic["largest_positive_delta"],
            "largest_negative_action": source_semantic["largest_negative_action"],
            "largest_negative_delta": source_semantic["largest_negative_delta"],
            **{f"delta_{action}": source_semantic["delta_by_action"][action] for action in ACTIONS},
        },
        "components": rows,
    }


def build_reason_prefix_patch_summary(
    probe: HFMechanismProbe,
    *,
    target_label: str,
    target_prompt: str,
    source_label: str,
    source_prompt: str,
    forced_action: str,
) -> dict:
    target_prefix = target_prompt + f'{{"action":"{forced_action}","reason":"'
    source_prefix = source_prompt + f'{{"action":"{forced_action}","reason":"'

    target_distribution = probe.next_token_distribution(target_prefix)
    source_distribution = probe.next_token_distribution(source_prefix)
    source_hidden_states = probe.get_action_prefix_hidden_states(source_prefix)

    baseline_distance = float(
        torch.abs(target_distribution["probs"] - source_distribution["probs"]).sum().item()
    )
    source_top_token = source_distribution["top_tokens"][0]["token_text"]
    source_top_token_id = source_distribution["top_tokens"][0]["token_id"]
    per_layer = []
    for layer_idx in range(len(probe.layers)):
        patched_distribution = probe.next_token_distribution(
            target_prefix,
            patch_layer_idx=layer_idx,
            patch_vector=source_hidden_states[layer_idx + 1][0, -1, :],
        )
        patched_distance = float(
            torch.abs(patched_distribution["probs"] - source_distribution["probs"]).sum().item()
        )
        source_top_prob = source_distribution["top_tokens"][0]["prob"]
        patched_source_top_prob = float(patched_distribution["probs"][source_top_token_id].item())
        per_layer.append(
            {
                "layer": layer_idx,
                "patched_top_token": patched_distribution["top_tokens"][0]["token_text"],
                "source_top_token": source_top_token,
                "patched_top_prob": patched_distribution["top_tokens"][0]["prob"],
                "source_top_prob_delta": patched_source_top_prob - source_top_prob,
                "distance_reduction": baseline_distance - patched_distance,
            }
        )

    ranked = sorted(per_layer, key=lambda row: row["distance_reduction"], reverse=True)
    return {
        "forced_action": forced_action,
        "target_label": target_label,
        "source_label": source_label,
        "target_baseline_top_tokens": target_distribution["top_tokens"],
        "source_baseline_top_tokens": source_distribution["top_tokens"],
        "baseline_distance_to_source": baseline_distance,
        "best_layers": ranked[:8],
        "all_layers": per_layer,
    }


def format_probability(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    return f"{value:.2e}"


def format_action_support_table(scored: dict[str, dict]) -> list[str]:
    lines = [
        "| Condition | Top Action | Top Prob | Margin vs 2nd | cooperate | defect | defend | negotiate | abstain |",
        "|-----------|------------|----------|---------------|-----------|--------|--------|-----------|---------|",
    ]
    for label, result in scored.items():
        probs = result["probabilities"]
        lines.append(
            f"| {label} | {result['top_action']} | {format_probability(result['top_probability'])} | "
            f"{format_probability(result['margin_vs_second'])} | {format_probability(probs['cooperate'])} | "
            f"{format_probability(probs['defect'])} | {format_probability(probs['defend'])} | "
            f"{format_probability(probs['negotiate'])} | {format_probability(probs['abstain'])} |"
        )
    return lines


def format_cosine_table(summary: dict[str, dict]) -> list[str]:
    lines = [
        "| Pair | layer_0 | layer_mid | layer_last | min_layer | min_cosine |",
        "|------|---------|-----------|------------|-----------|------------|",
    ]
    for pair_name, values in summary.items():
        lines.append(
            f"| {pair_name} | {format_probability(values['layer_0'])} | "
            f"{format_probability(values['layer_mid'])} | {format_probability(values['layer_last'])} | "
            f"{values['min_layer']} | {format_probability(values['min_cosine'])} |"
        )
    return lines


def format_trace_section(traces: dict[str, dict], max_steps: int = 6) -> list[str]:
    lines = []
    for label, trace in traces.items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- decoded suffix: `{trace['decoded_suffix']}`")
        if not trace["steps"]:
            lines.append("- no decode steps captured")
        else:
            for idx, step in enumerate(trace["steps"][:max_steps], start=1):
                lines.append(
                    f"- step {idx}: next=`{step['next_token_text']}` "
                    f"top1=`{step['top_tokens'][0]['token_text']}`"
                )
        lines.append("")
    return lines


def format_forced_trace_section(forced_traces: dict[str, dict[str, dict]], max_steps: int = 6) -> list[str]:
    lines = []
    for action_name, traces in forced_traces.items():
        lines.append(f"### forced `{action_name}` reason trace")
        lines.append("")
        lines.extend(format_trace_section(traces, max_steps=max_steps))
    return lines


def format_activation_patch_section(patch_results: dict[str, dict]) -> list[str]:
    lines = []
    for patch_name, result in patch_results.items():
        source = result["source_label"]
        target = result["target_label"]
        lines.extend(
            [
                f"### {patch_name}",
                "",
                f"- target baseline top action: `{result['target_baseline']['top_action']}`",
                f"- source baseline top action: `{result['source_baseline']['top_action']}`",
                f"- baseline L1 distance to source: `{result['baseline_distance_to_source']:.3f}`",
                "",
                "| layer | patched top | cooperate | defect | negotiate | source-top delta | distance reduction |",
                "|------|-------------|-----------|--------|-----------|------------------|--------------------|",
            ]
        )
        for row in result["best_layers"]:
            lines.append(
                f"| {row['layer']} | {row['patched_top_action']} | {format_probability(row['cooperate'])} | "
                f"{format_probability(row['defect'])} | {format_probability(row['negotiate'])} | "
                f"{row['source_top_action_delta']:.3f} | {row['distance_reduction']:.3f} |"
            )
        lines.append("")
    return lines


def format_multi_position_patch_section(patch_results: dict[str, dict]) -> list[str]:
    lines = []
    for patch_name, result in patch_results.items():
        lines.extend(
            [
                f"### {patch_name}",
                "",
                f"- target baseline top action: `{result['target_baseline']['top_action']}`",
                f"- source baseline top action: `{result['source_baseline']['top_action']}`",
                f"- baseline L1 distance to source: `{result['baseline_distance_to_source']:.3f}`",
                f"- spans tested: `{', '.join(f'last_{span}' for span in result['span_sizes'])}`",
                "",
                "| span | layer | patched top | cooperate | defect | negotiate | source-top delta | distance reduction |",
                "|------|-------|-------------|-----------|--------|-----------|------------------|--------------------|",
            ]
        )
        for row in result["best_combinations"]:
            lines.append(
                f"| {row['span_label']} | {row['layer']} | {row['patched_top_action']} | "
                f"{format_probability(row['cooperate'])} | {format_probability(row['defect'])} | "
                f"{format_probability(row['negotiate'])} | {row['source_top_action_delta']:.3f} | "
                f"{row['distance_reduction']:.3f} |"
            )
        lines.append("")
    return lines


def format_attention_head_patch_section(patch_results: dict[str, dict]) -> list[str]:
    lines = []
    for patch_name, result in patch_results.items():
        lines.extend(
            [
                f"### {patch_name}",
                "",
                f"- target baseline top action: `{result['target_baseline']['top_action']}`",
                f"- source baseline top action: `{result['source_baseline']['top_action']}`",
                f"- baseline L1 distance to source: `{result['baseline_distance_to_source']:.3f}`",
                f"- candidate layers: `{', '.join(str(layer) for layer in result['candidate_layers'])}`",
                f"- heads per layer: `{result['num_heads']}` (head dim `{result['head_dim']}`)",
                "",
                "| layer | head | patched top | cooperate | defect | negotiate | source-top delta | distance reduction |",
                "|------|------|-------------|-----------|--------|-----------|------------------|--------------------|",
            ]
        )
        for row in result["best_heads"]:
            lines.append(
                f"| {row['layer']} | {row['head']} | {row['patched_top_action']} | "
                f"{format_probability(row['cooperate'])} | {format_probability(row['defect'])} | "
                f"{format_probability(row['negotiate'])} | {row['source_top_action_delta']:.3f} | "
                f"{row['distance_reduction']:.3f} |"
            )
        lines.append("")
    return lines


def format_attention_head_set_patch_section(patch_results: dict[str, dict]) -> list[str]:
    lines = []
    for patch_name, result in patch_results.items():
        lines.extend(
            [
                f"### {patch_name}",
                "",
                f"- target baseline top action: `{result['target_baseline']['top_action']}`",
                f"- source baseline top action: `{result['source_baseline']['top_action']}`",
                f"- baseline L1 distance to source: `{result['baseline_distance_to_source']:.3f}`",
                "",
                "| head set | members | patched top | cooperate | defect | negotiate | source-top delta | distance reduction |",
                "|----------|---------|-------------|-----------|--------|-----------|------------------|--------------------|",
            ]
        )
        for row in result["best_sets"]:
            members = ", ".join(f"L{layer}H{head}" for layer, head in row["members"])
            lines.append(
                f"| {row['set_label']} | `{members}` | {row['patched_top_action']} | "
                f"{format_probability(row['cooperate'])} | {format_probability(row['defect'])} | "
                f"{format_probability(row['negotiate'])} | {row['source_top_action_delta']:.3f} | "
                f"{row['distance_reduction']:.3f} |"
            )
        lines.append("")
    return lines


def format_head_overlap_section(overlap_results: dict) -> list[str]:
    lines = [
        "| k | intersection | jaccard | overlapping heads |",
        "|---|--------------|---------|-------------------|",
    ]
    for row in overlap_results["overlaps"]:
        members = ", ".join(f"L{layer}H{head}" for layer, head in row["overlap_members"]) or "-"
        lines.append(
            f"| {row['k']} | {row['intersection_count']} | {row['jaccard']:.3f} | `{members}` |"
        )
    return lines


def format_mlp_patch_section(patch_results: dict[str, dict]) -> list[str]:
    lines = []
    for patch_name, result in patch_results.items():
        lines.extend(
            [
                f"### {patch_name}",
                "",
                f"- target baseline top action: `{result['target_baseline']['top_action']}`",
                f"- source baseline top action: `{result['source_baseline']['top_action']}`",
                f"- baseline L1 distance to source: `{result['baseline_distance_to_source']:.3f}`",
                f"- candidate layers: `{', '.join(str(layer) for layer in result['candidate_layers'])}`",
                "",
                "| layer | patched top | cooperate | defect | negotiate | source-top delta | distance reduction |",
                "|------|-------------|-----------|--------|-----------|------------------|--------------------|",
            ]
        )
        for row in result["best_layers"]:
            lines.append(
                f"| {row['layer']} | {row['patched_top_action']} | {format_probability(row['cooperate'])} | "
                f"{format_probability(row['defect'])} | {format_probability(row['negotiate'])} | "
                f"{row['source_top_action_delta']:.3f} | {row['distance_reduction']:.3f} |"
            )
        lines.append("")
    return lines


def format_semantic_direction_section(semantic_results: dict[str, dict]) -> list[str]:
    lines = []
    for patch_name, result in semantic_results.items():
        source_shift = result["source_shift"]
        lines.extend(
            [
                f"### {patch_name}",
                "",
                f"- source shift: `{source_shift['largest_positive_action']}` {source_shift['largest_positive_delta']:.3f}, "
                f"`{source_shift['largest_negative_action']}` {source_shift['largest_negative_delta']:.3f}",
                "",
                "| component | distance reduction | cosine to source shift | main gain | main loss | d(cooperate) | d(defect) | d(defend) | d(negotiate) | d(abstain) |",
                "|-----------|--------------------|------------------------|-----------|-----------|--------------|-----------|-----------|--------------|------------|",
            ]
        )
        for row in result["components"]:
            gain = f"{row['largest_positive_action']} {row['largest_positive_delta']:.3f}"
            loss = f"{row['largest_negative_action']} {row['largest_negative_delta']:.3f}"
            lines.append(
                f"| {row['component']} | {row['distance_reduction']:.3f} | {row['direction_cosine_to_source']:.3f} | "
                f"{gain} | {loss} | {row['delta_cooperate']:.3f} | {row['delta_defect']:.3f} | "
                f"{row['delta_defend']:.3f} | {row['delta_negotiate']:.3f} | {row['delta_abstain']:.3f} |"
            )
        lines.append("")
    return lines


def format_reason_patch_section(reason_patch_results: dict[str, dict]) -> list[str]:
    lines = []
    for patch_name, result in reason_patch_results.items():
        lines.extend(
            [
                f"### {patch_name}",
                "",
                f"- forced action: `{result['forced_action']}`",
                f"- target baseline top token: `{result['target_baseline_top_tokens'][0]['token_text']}`",
                f"- source baseline top token: `{result['source_baseline_top_tokens'][0]['token_text']}`",
                f"- baseline L1 distance to source: `{result['baseline_distance_to_source']:.3f}`",
                "",
                "| layer | patched top token | patched top prob | source-top delta | distance reduction |",
                "|------|-------------------|------------------|------------------|--------------------|",
            ]
        )
        for row in result["best_layers"]:
            lines.append(
                f"| {row['layer']} | `{row['patched_top_token']}` | {format_probability(row['patched_top_prob'])} | "
                f"{row['source_top_prob_delta']:.3f} | {row['distance_reduction']:.3f} |"
            )
        lines.append("")
    return lines


def build_markdown(
    *,
    model_id: str,
    transcript_results: dict | None,
    wording_results: dict | None,
) -> str:
    lines = [
        "# GPU Mechanism Pilot Summary",
        "",
        f"Model: `{model_id}`",
        "",
        "This formal GPU pass can include up to two probes:",
        "- transcript-conditioned action support and hidden-state divergence",
        "- wording-conditioned later-decoding analysis for the `single_norm` anomaly",
        "",
    ]

    if transcript_results is not None:
        lines.extend(
            [
                "## 1. Phi Transcript Probe",
                "",
                "Action support at the JSON action prefix:",
                "",
            ]
        )
        lines.extend(format_action_support_table(transcript_results["action_probe"]))
        lines.extend(
            [
                "",
                "Layerwise cosine summary at the action prefix:",
                "",
            ]
        )
        lines.extend(format_cosine_table(transcript_results["pairwise_cosines"]))
        lines.extend(["", "Short decode traces:", ""])
        lines.extend(format_trace_section(transcript_results["greedy_traces"]))

    if wording_results is not None:
        lines.extend(
            [
                "## 2. Single-Norm Wording Probe",
                "",
                "Action support at the JSON action prefix:",
                "",
            ]
        )
        lines.extend(format_action_support_table(wording_results["action_probe"]))
        lines.extend(
            [
                "",
                "Cosine summary at the action prefix:",
                "",
            ]
        )
        lines.extend(format_cosine_table(wording_results["action_prefix_cosines"]))
        lines.extend(
            [
                "",
                "Cosine summary after forcing `{\"action\":\"cooperate\",\"reason\":\"`:",
                "",
            ]
        )
        lines.extend(format_cosine_table(wording_results["reason_prefix_cosines"]))
        lines.extend(["", "Later-decoding traces from the action prefix:", ""])
        lines.extend(format_trace_section(wording_results["greedy_traces"]))
        lines.extend(["", "Forced continuation traces:", ""])
        lines.extend(format_forced_trace_section(wording_results["forced_traces"]))
        lines.extend(["", "Activation patching at the action prefix:", ""])
        lines.extend(format_activation_patch_section(wording_results["activation_patching"]))
        lines.extend(["", "Multi-position residual patching at the action prefix:", ""])
        lines.extend(format_multi_position_patch_section(wording_results["multi_position_patching"]))
        lines.extend(["", "Attention-head patching at the action prefix:", ""])
        lines.extend(format_attention_head_patch_section(wording_results["attention_head_patching"]))
        lines.extend(["", "Head-set patching at the action prefix:", ""])
        lines.extend(format_attention_head_set_patch_section(wording_results["attention_head_set_patching"]))
        lines.extend(["", "Top-head overlap across wording lines:", ""])
        lines.extend(format_head_overlap_section(wording_results["head_overlap"]))
        lines.extend(["", "Late-layer MLP patching at the action prefix:", ""])
        lines.extend(format_mlp_patch_section(wording_results["mlp_patching"]))
        lines.extend(["", "Semantic action-direction follow-up:", ""])
        lines.extend(format_semantic_direction_section(wording_results["semantic_directions"]))
        lines.extend(["", "Activation patching at forced reason prefixes:", ""])
        lines.extend(format_reason_patch_section(wording_results["reason_prefix_patching"]))

    lines.extend(
        [
            "## 3. Reading",
            "",
            "- The transcript probe shows how cue polarity changes both local action support and later-layer state geometry.",
            "- The wording probe, when enabled, tests whether the anomaly lives at action selection, later decoding, or both.",
            "- The action-prefix patching tables show which late layers move `soft_default` toward the cleaner wording variants.",
            "- The multi-position and head patching tables test whether that effect stays local to one residual slot or concentrates in specific attention heads.",
            "- The head-set patching tables test whether a sparse subset of heads can reconstruct most of the late-layer residual effect.",
            "- The overlap and MLP tables test whether the key heads align across phrasing variants and how much of the same effect bypasses attention through late MLP blocks.",
            "- The semantic follow-up tables show which action-support direction those load-bearing components actually push.",
            "- The forced reason-prefix patching tables test whether the same late layers also control the first reason token once the action is held fixed.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_wording_analysis(
    probe: HFMechanismProbe,
    prompts: dict[str, str],
    max_new_tokens: int,
) -> dict:
    action_probe = {
        label: probe.score_actions(prompt)
        for label, prompt in prompts.items()
    }
    action_prefix_vectors = {
        label: result["layer_vectors"]
        for label, result in action_probe.items()
    }
    reason_prefix_vectors = {
        label: probe.extract_layer_vectors(prompt + '{"action":"cooperate","reason":"')
        for label, prompt in prompts.items()
    }
    traces = {
        label: probe.greedy_trace(prompt, max_new_tokens=max_new_tokens)
        for label, prompt in prompts.items()
    }
    forced_traces = {
        action_name: {
            label: probe.forced_continuation_trace(prompt, action_name, max_new_tokens=max_new_tokens)
            for label, prompt in prompts.items()
        }
        for action_name in ("defect", "negotiate")
    }
    activation_patching = {
        "positive_reframe_to_soft_default": build_activation_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="positive_reframe",
            source_prompt=prompts["positive_reframe"],
        ),
        "forbidden_hard_to_soft_default": build_activation_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="forbidden_hard",
            source_prompt=prompts["forbidden_hard"],
        ),
    }
    multi_position_patching = {
        "positive_reframe_to_soft_default": build_multi_position_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="positive_reframe",
            source_prompt=prompts["positive_reframe"],
        ),
        "forbidden_hard_to_soft_default": build_multi_position_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="forbidden_hard",
            source_prompt=prompts["forbidden_hard"],
        ),
    }
    attention_head_patching = {
        "positive_reframe_to_soft_default": build_attention_head_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="positive_reframe",
            source_prompt=prompts["positive_reframe"],
            candidate_layers=select_candidate_layers(
                activation_patching["positive_reframe_to_soft_default"]["best_layers"]
            ),
        ),
        "forbidden_hard_to_soft_default": build_attention_head_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="forbidden_hard",
            source_prompt=prompts["forbidden_hard"],
            candidate_layers=select_candidate_layers(
                activation_patching["forbidden_hard_to_soft_default"]["best_layers"]
            ),
        ),
    }
    attention_head_set_patching = {
        patch_name: build_attention_head_set_patch_summary(
            probe,
            target_label=result["target_label"],
            target_prompt=prompts[result["target_label"]],
            source_label=result["source_label"],
            source_prompt=prompts[result["source_label"]],
            base_head_summary=result,
        )
        for patch_name, result in attention_head_patching.items()
    }
    head_overlap = compute_head_overlap_summary(
        attention_head_patching["positive_reframe_to_soft_default"]["all_heads"],
        attention_head_patching["forbidden_hard_to_soft_default"]["all_heads"],
    )
    mlp_patching = {
        "positive_reframe_to_soft_default": build_mlp_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="positive_reframe",
            source_prompt=prompts["positive_reframe"],
            candidate_layers=select_candidate_layers(
                activation_patching["positive_reframe_to_soft_default"]["best_layers"]
            ),
        ),
        "forbidden_hard_to_soft_default": build_mlp_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="forbidden_hard",
            source_prompt=prompts["forbidden_hard"],
            candidate_layers=select_candidate_layers(
                activation_patching["forbidden_hard_to_soft_default"]["best_layers"]
            ),
        ),
    }
    semantic_directions = {
        "positive_reframe_to_soft_default": build_semantic_direction_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="positive_reframe",
            source_prompt=prompts["positive_reframe"],
            activation_summary=activation_patching["positive_reframe_to_soft_default"],
            head_set_summary=attention_head_set_patching["positive_reframe_to_soft_default"],
            mlp_summary=mlp_patching["positive_reframe_to_soft_default"],
        ),
        "forbidden_hard_to_soft_default": build_semantic_direction_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="forbidden_hard",
            source_prompt=prompts["forbidden_hard"],
            activation_summary=activation_patching["forbidden_hard_to_soft_default"],
            head_set_summary=attention_head_set_patching["forbidden_hard_to_soft_default"],
            mlp_summary=mlp_patching["forbidden_hard_to_soft_default"],
        ),
    }
    reason_prefix_patching = {
        "defect_positive_reframe_to_soft_default": build_reason_prefix_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="positive_reframe",
            source_prompt=prompts["positive_reframe"],
            forced_action="defect",
        ),
        "defect_forbidden_hard_to_soft_default": build_reason_prefix_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="forbidden_hard",
            source_prompt=prompts["forbidden_hard"],
            forced_action="defect",
        ),
        "negotiate_positive_reframe_to_soft_default": build_reason_prefix_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="positive_reframe",
            source_prompt=prompts["positive_reframe"],
            forced_action="negotiate",
        ),
        "negotiate_forbidden_hard_to_soft_default": build_reason_prefix_patch_summary(
            probe,
            target_label="soft_default",
            target_prompt=prompts["soft_default"],
            source_label="forbidden_hard",
            source_prompt=prompts["forbidden_hard"],
            forced_action="negotiate",
        ),
    }
    return {
        "prompts": prompts,
        "action_probe": {
            label: {k: v for k, v in result.items() if k != "layer_vectors"}
            for label, result in action_probe.items()
        },
        "action_prefix_cosines": summarize_layer_cosines(action_prefix_vectors),
        "reason_prefix_cosines": summarize_layer_cosines(reason_prefix_vectors),
        "greedy_traces": traces,
        "forced_traces": forced_traces,
        "activation_patching": activation_patching,
        "multi_position_patching": multi_position_patching,
        "attention_head_patching": attention_head_patching,
        "attention_head_set_patching": attention_head_set_patching,
        "head_overlap": head_overlap,
        "mlp_patching": mlp_patching,
        "semantic_directions": semantic_directions,
        "reason_prefix_patching": reason_prefix_patching,
    }


def build_transcript_analysis(
    probe: HFMechanismProbe,
    prompts: dict[str, str],
    max_new_tokens: int,
) -> dict:
    action_probe = {
        label: probe.score_actions(prompt)
        for label, prompt in prompts.items()
    }
    prefix_vectors = {
        label: result["layer_vectors"]
        for label, result in action_probe.items()
    }
    traces = {
        label: probe.greedy_trace(prompt, max_new_tokens=max_new_tokens)
        for label, prompt in prompts.items()
    }
    return {
        "prompts": prompts,
        "action_probe": {
            label: {k: v for k, v in result.items() if k != "layer_vectors"}
            for label, result in action_probe.items()
        },
        "pairwise_cosines": summarize_layer_cosines(prefix_vectors),
        "greedy_traces": traces,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-backed Phi mechanism probes")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-basename", default="mechanism_pilot_gpu_phi3")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--probe-mode",
        choices=["full", "transcript-only", "wording-only"],
        default="full",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    probe = HFMechanismProbe(
        args.model_id,
        args.device,
        local_files_only=args.local_files_only,
    )

    transcript_results = None
    wording_results = None
    if args.probe_mode in {"full", "transcript-only"}:
        transcript_results = build_transcript_analysis(
            probe,
            build_transcript_prompts(),
            max_new_tokens=args.max_new_tokens,
        )
    if args.probe_mode in {"full", "wording-only"}:
        wording_results = build_wording_analysis(
            probe,
            build_wording_prompts(),
            max_new_tokens=args.max_new_tokens,
        )

    payload = {
        "model_id": args.model_id,
        "probe_mode": args.probe_mode,
        "transcript_probe": transcript_results,
        "wording_probe": wording_results,
    }

    json_path = RESULTS_DIR / f"{args.output_basename}.json"
    md_path = RESULTS_DIR / f"{args.output_basename}_summary.md"
    write_text(json_path, json.dumps(payload, indent=2, ensure_ascii=False))
    write_text(
        md_path,
        build_markdown(
            model_id=args.model_id,
            transcript_results=transcript_results,
            wording_results=wording_results,
        ),
    )
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
