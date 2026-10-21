"""Cliente local do modelo YOLO."""

from pathlib import Path
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError

from fitpose_ai.domain.entities import BoundingBox, Detection, InferenceResult


class YoloDetector:
    """Cliente que converte a saída do Ultralytics para o domínio."""

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em '{self._model_path}'.")
        from ultralytics import YOLO
        if self._model is None:
            self._model = YOLO(str(self._model_path))
        return self._model

    def detect(self, image: bytes) -> InferenceResult:
        try:
            decoded_image = Image.open(BytesIO(image)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("O arquivo enviado não é uma imagem válida.") from exc

        results = self._load_model().predict(
            source=decoded_image,
            conf=self._confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                coordinates = [max(0, int(value)) for value in box.tolist()]
                index = int(class_id.item())
                detections.append(Detection(
                    class_name=str(result.names[index]),
                    confidence=round(float(confidence.item()), 4),
                    box=BoundingBox(coordinates[0], coordinates[1], coordinates[2], coordinates[3]),
                ))
        return InferenceResult(tuple(detections))
