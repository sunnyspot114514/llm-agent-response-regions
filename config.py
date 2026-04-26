"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    path: str
    role: str
    default_temperature: float = 0.7
    backend: str = "llama_cpp"
    ollama_model: Optional[str] = None


class InferenceConfig(BaseModel):
    n_ctx: int = 2048
    n_threads: int = 8
    seed: int = 42


class AgentConfig(BaseModel):
    id: str
    model: str
    temperature: Optional[float] = None


class ExperimentConfig(BaseModel):
    agents: list[AgentConfig]
    forbidden: list[str] = Field(default_factory=list)
    rounds: int = 30
    norm_mode: str = "soft"
    norm_rules: list[str] = Field(default_factory=list)
    social_exposure: Optional[dict] = None
    prompt_config: Optional[dict] = None
    observation_config: Optional[dict] = None


class Settings:
    """Load and expose config files under ``configs/``."""

    def __init__(self, config_dir: Path = Path("configs")):
        self.config_dir = config_dir
        self._models: dict[str, ModelConfig] = {}
        self._inference: InferenceConfig = InferenceConfig()
        self._experiments: dict[str, ExperimentConfig] = {}
        self._defaults: dict = {}
        self._action_space: list[str] = []
        self._logs_dir: Path = Path("logs")

        self._load_configs()

    def _load_configs(self):
        models_file = self.config_dir / "models.yaml"
        if models_file.exists():
            data = yaml.safe_load(models_file.read_text(encoding="utf-8"))
            for name, cfg in data.get("models", {}).items():
                self._models[name] = ModelConfig(**cfg)
            if "inference" in data:
                self._inference = InferenceConfig(**data["inference"])

        exp_file = self.config_dir / "experiment.yaml"
        if exp_file.exists():
            data = yaml.safe_load(exp_file.read_text(encoding="utf-8"))
            self._defaults = data.get("defaults", {})
            self._action_space = data.get("action_space", {}).get("allowed", [])
            self._logs_dir = Path(data.get("logs_dir", "logs"))

            for name, cfg in data.get("experiments", {}).items():
                agents = [AgentConfig(**agent_cfg) for agent_cfg in cfg.get("agents", [])]
                self._experiments[name] = ExperimentConfig(
                    agents=agents,
                    forbidden=cfg.get("forbidden", []),
                    rounds=cfg.get("rounds", self._defaults.get("rounds", 30)),
                    norm_mode=cfg.get("norm_mode", "soft"),
                    norm_rules=cfg.get("norm_rules", []),
                    social_exposure=cfg.get("social_exposure"),
                    prompt_config=cfg.get("prompt_config"),
                    observation_config=cfg.get("observation_config"),
                )

    def get_model(self, name: str) -> ModelConfig:
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        return self._models[name]

    def get_model_path(self, name: str) -> str:
        return self.get_model(name).path

    def get_experiment(self, name: str) -> ExperimentConfig:
        if name not in self._experiments:
            raise ValueError(f"Unknown experiment: {name}")
        return self._experiments[name]

    def list_experiments(self) -> list[str]:
        return list(self._experiments.keys())

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    @property
    def inference(self) -> InferenceConfig:
        return self._inference

    @property
    def defaults(self) -> dict:
        return self._defaults

    @property
    def action_space(self) -> list[str]:
        return self._action_space

    @property
    def logs_dir(self) -> Path:
        return self._logs_dir


settings = Settings()


def load_config() -> dict:
    """Load the raw config dict for batch-oriented scripts."""

    config_dir = Path("configs")
    result = {
        "models": {},
        "experiments": {},
        "runtime": {},
    }

    models_file = config_dir / "models.yaml"
    if models_file.exists():
        data = yaml.safe_load(models_file.read_text(encoding="utf-8"))
        result["models"] = data.get("models", {})
        if "inference" in data:
            result["runtime"]["n_threads"] = data["inference"].get("n_threads", 6)

    exp_file = config_dir / "experiment.yaml"
    if exp_file.exists():
        data = yaml.safe_load(exp_file.read_text(encoding="utf-8"))
        result["experiments"] = data.get("experiments", {})
        result["defaults"] = data.get("defaults", {})
        result["action_space"] = data.get("action_space", {})
        result["logs_dir"] = data.get("logs_dir", "logs")

    return result
