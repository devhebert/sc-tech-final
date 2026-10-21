from typing import Protocol

from .entities import InferenceResult


class ExerciseDetector(Protocol):
    def detect(self, image: bytes) -> InferenceResult:
        """Executa uma inferência sobre os bytes de uma imagem."""
