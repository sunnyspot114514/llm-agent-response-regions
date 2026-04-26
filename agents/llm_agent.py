"""LLM-based agent using llama-cpp-python."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from llama_cpp import Llama
from loguru import logger

from .base_agent import BaseAgent
from .output_protocol import get_action_output_grammar, parse_action_output
from schemas.action import Action, ActionSpace
from schemas.world_state import WorldState


class LLMAgent(BaseAgent):
    """鍩轰簬鏈湴 LLM 鐨?Agent"""

    def __init__(
        self,
        agent_id: str,
        model_path: str,
        action_space: Optional[ActionSpace] = None,
        n_ctx: int = 2048,
        n_threads: int = 8,
        temperature: float = 0.7,
        seed: int = 42,
        prompt_config: Optional[dict] = None,
    ):
        super().__init__(agent_id, action_space, prompt_config=prompt_config)

        self.model_path = model_path
        self.temperature = temperature
        self.seed = seed

        logger.info(f"Loading model for {agent_id}: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            seed=seed,
            verbose=False,
        )

        self.state.model_name = Path(model_path).stem
        logger.info(f"Agent {agent_id} ready")

    def _build_prompt(
        self,
        world_state: WorldState,
        other_agents: list[BaseAgent],
        last_round_actions: Optional[dict[str, Action]] = None,
    ) -> str:
        """鏋勫缓鍐崇瓥 prompt."""

        allowed = ", ".join(self.action_space.allowed)
        forbidden = ", ".join(self.action_space.forbidden) if self.action_space.forbidden else "鏃?"

        if last_round_actions:
            last_actions_str = "\n".join(
                f"- {agent_id}: {action.action_type}"
                for agent_id, action in last_round_actions.items()
            )
        else:
            last_actions_str = "杩欐槸绗竴杞?"

        others_str = ", ".join(agent.agent_id for agent in other_agents) if other_agents else "鏃?"

        return f"""<|im_start|>system
浣犳槸涓€涓弬涓庡浜哄崥寮堢殑 Agent锛孖D 鏄?{self.agent_id}銆?
浣犻渶瑕佹牴鎹綋鍓嶇姸鎬侀€夋嫨涓€涓涓恒€?

鍙€夎涓? {allowed}
绂佹琛屼负: {forbidden}

鍙兘杩斿洖涓€涓?JSON 瀵硅薄锛屼笉瑕佹湁浠讳綍棰濆鏂囧瓧:
{{"action": "cooperate", "reason": "鐞嗙敱"}}
<|im_end|>
<|im_start|>user
{world_state.to_prompt()}

鍏朵粬鍙備笌鑰? {others_str}

涓婁竴杞涓?
{last_actions_str}

浣犵殑璧勬簮: {self.state.resources}
浣犵殑澹拌獕: {self.state.reputation:.2f}

璇烽€夋嫨浣犵殑琛屼负:
<|im_end|>
<|im_start|>assistant
"""

    def decide(
        self,
        world_state: WorldState,
        other_agents: list[BaseAgent],
        last_round_actions: Optional[dict[str, Action]] = None,
        observation_context: Optional[dict] = None,
    ) -> tuple[Action, dict]:
        """LLM 鍐崇瓥"""

        prompt = self._build_prompt(world_state, other_agents, last_round_actions)
        grammar = get_action_output_grammar(tuple(self.action_space.allowed))

        start_time = time.time()
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=self.temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
            grammar=grammar,
        )

        inference_time = int((time.time() - start_time) * 1000)
        raw_output = output["choices"][0]["text"].strip()
        action, parse_metadata = parse_action_output(raw_output, self.action_space.allowed)

        if action is None:
            logger.warning(f"Agent {self.agent_id} parse failed, defaulting to abstain")
            action = Action(action_type="abstain", reason="parse_failed")

        metadata = {
            "raw_output": raw_output,
            "think_content": parse_metadata.get("think_content"),
            "load_time_ms": 0,
            "infer_time_ms": inference_time,
            "parser_status": parse_metadata.get("parser_status"),
            "prompt_echo_detected": parse_metadata.get("prompt_echo_detected", False),
            "output_has_extra_text": parse_metadata.get("output_has_extra_text", False),
            "grammar_enforced": True,
        }

        logger.debug(
            f"Agent {self.agent_id} output ({inference_time}ms, {metadata['parser_status']}): "
            f"{raw_output[:80]}"
        )
        return action, metadata

    def __del__(self):
        if hasattr(self, "llm"):
            del self.llm
