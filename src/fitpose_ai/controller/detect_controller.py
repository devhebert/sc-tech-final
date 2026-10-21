"""Controller HTTP do caso de uso de detecção."""

from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from fitpose_ai.application.detect_exercise import DetectExerciseUseCase
from fitpose_ai.domain.entities import InferenceResult


class BoundingBoxResponse(BaseModel):
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    x_max: int = Field(ge=0)
    y_max: int = Field(ge=0)


class DetectionResponse(BaseModel):
    class_name: str
    label: str
    confidence: float = Field(ge=0, le=1)
    box: BoundingBoxResponse


class InferenceResponse(BaseModel):
    status: str
    detections: list[DetectionResponse]


def create_app(use_case: DetectExerciseUseCase, model_path: str) -> FastAPI:
    app = FastAPI(title="FitPose AI", description="Detecção de exercícios com YOLO.", version="1.0.0")

    @app.get("/health", tags=["Sistema"])
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_path}

    @app.post("/detect", response_model=InferenceResponse, tags=["Inferência"])
    async def detect(file: Annotated[UploadFile, File(description="Imagem do exercício")]) -> InferenceResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Envie um arquivo de imagem.")
        image = await file.read()
        try:
            result = use_case.execute(image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Não foi possível processar a imagem: {exc}") from exc
        return _to_response(result)

    return app


def _to_response(result: InferenceResult) -> InferenceResponse:
    return InferenceResponse(status="success", detections=[
        DetectionResponse(
            class_name=item.class_name,
            label=_friendly_label(item.class_name),
            confidence=item.confidence,
            box=BoundingBoxResponse(
                x_min=item.box.x_min,
                y_min=item.box.y_min,
                x_max=item.box.x_max,
                y_max=item.box.y_max,
            ),
        ) for item in result.detections
    ])


def _friendly_label(class_name: str) -> str:
    labels: dict[str, str] = {
        "Normal_BrenchPress": "Supino",
        "Normal_Deadlifts": "Levantamento terra",
        "Normal_Dips": "Paralelas (Dips)",
        "Normal_DumbbellBicepCurls": "Rosca bíceps com halteres",
        "Normal_JumpingJacks": "Polichinelo",
        "Normal_LatPulldown": "Puxada alta",
        "Normal_LegPress": "Leg press",
        "Normal_Lunges": "Avanço",
        "Normal_MountainClimbers": "Escalador",
        "Normal_OverheadPress": "Desenvolvimento de ombros",
        "Normal_Plunk": "Prancha",
        "Normal_PullUps": "Barra fixa",
        "Normal_PushUps": "Flexão de braço",
        "Normal_RowingMachineExercise": "Remada na máquina",
        "Normal_SquatsWithBarbell": "Agachamento com barra",
    }
    return labels.get(class_name, class_name)
