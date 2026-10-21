from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Detection, ...]
