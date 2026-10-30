"""Treina múltiplas variantes YOLO e salva seus melhores pesos."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def main() -> None:
    os.chdir(PROJECT_ROOT)

    dataset_value = os.environ.get("DATASET", "data/fitness_normal/data.yaml")
    dataset = Path(dataset_value)
    if not dataset.is_absolute():
        dataset = PROJECT_ROOT / dataset
    if not dataset.is_file():
        raise SystemExit(f"Dataset não encontrado: {dataset_value}")

    models_value = os.environ.get("MODELS", "yolov8n.pt yolov8l.pt")
    models = models_value.split()
    if not models:
        raise SystemExit("Nenhum modelo foi configurado em MODELS.")

    epochs = env_int("EPOCHS", 100)
    patience = env_int("PATIENCE", 20)
    imgsz = env_int("IMGSZ", 512)
    batch = env_int("BATCH", 4)
    workers = env_int("WORKERS", 2)
    run_prefix = os.environ.get("RUN_PREFIX", "fitpose")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        model_stem = Path(model_name).stem
        run_name = f"{run_prefix}-{model_stem}"
        output_path = MODELS_DIR / f"{model_stem}.pt"
        best_path = RUNS_DIR / run_name / "weights" / "best.pt"

        print(f"Treinando: {model_name} -> runs/{run_name}")
        model = YOLO(model_name)
        model.train(
            data=str(dataset),
            epochs=epochs,
            patience=patience,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            cache=False,
            amp=True,
            project=str(RUNS_DIR),
            name=run_name,
        )

        if not best_path.is_file():
            raise SystemExit(f"Checkpoint não encontrado: {best_path}")

        shutil.copy2(best_path, output_path)
        print(f"Modelo pronto em: {PROJECT_ROOT / 'models' / output_path.name}")


if __name__ == "__main__":
    main()
