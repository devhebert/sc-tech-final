from dataclasses import dataclass

from fitpose_ai.domain.entities import InferenceResult
from fitpose_ai.domain.ports import ExerciseDetector


@dataclass(frozen=True)
class DetectExerciseUseCase:
    detector: ExerciseDetector

    def execute(self, image: bytes) -> InferenceResult:
        if len(image) == 0:
            raise ValueError("A imagem não pode estar vazia.")
        return self.detector.detect(image)
