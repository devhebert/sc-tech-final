import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    path: Path = Path("models/best-v8.pt")
    confidence_threshold: float = 0.50


@dataclass(frozen=True)
class ApplicationConfig:
    model: ModelConfig

    @staticmethod
    def from_environment() -> "ApplicationConfig":
        model_path = Path(os.getenv("MODEL_PATH", "models/best-v8.pt"))
        confidence = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
        return ApplicationConfig(ModelConfig(path=model_path, confidence_threshold=confidence))
