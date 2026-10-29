"""Treina o modelo base YOLOv8 e copia o melhor peso para models/best.pt."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "fitness_normal" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    os.chdir(PROJECT_ROOT)
    if not DATASET.is_file():
        raise SystemExit(f"Dataset não encontrado: {DATASET.relative_to(PROJECT_ROOT)}")

    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.train(
        data=str(DATASET),
        epochs=50,
        imgsz=640,
        project=str(RUNS_DIR),
        name="fitpose-yolov8",
    )

    best_models = [
        path
        for path in RUNS_DIR.glob("**/weights/best.pt")
        if path.is_file()
    ]
    if not best_models:
        raise SystemExit("Nenhum best.pt foi encontrado em runs/.")

    latest_best = max(best_models, key=lambda path: path.stat().st_mtime)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / "best.pt"
    shutil.copy2(latest_best, destination)
    print(f"Modelo selecionado: {latest_best.relative_to(PROJECT_ROOT)}")
    print(f"Modelo pronto em: {destination}")


if __name__ == "__main__":
    main()
