from fastapi import FastAPI

from fitpose_ai.application.detect_exercise import DetectExerciseUseCase
from fitpose_ai.client.yolo_client import YoloDetector
from fitpose_ai.config.settings import ApplicationConfig
from fitpose_ai.controller.detect_controller import create_app


def build_application(config: ApplicationConfig | None = None) -> FastAPI:
    application_config = config or ApplicationConfig.from_environment()
    detector = YoloDetector(application_config.model.path, application_config.model.confidence_threshold)
    use_case = DetectExerciseUseCase(detector)
    return create_app(use_case, str(application_config.model.path))
