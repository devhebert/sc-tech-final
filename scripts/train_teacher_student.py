"""Gera pseudo-rótulos e treina o modelo aluno."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRA_IMAGES = PROJECT_ROOT / "data" / "fitness_extra" / "images"
DEFAULT_TEACHER_DATA = PROJECT_ROOT / "data" / "fitness_teacher"


def environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def main() -> None:
    os.chdir(PROJECT_ROOT)
    teacher_model = os.environ.get("TEACHER_MODEL", "yolo11l.pt")
    extra_images = environment_path("EXTRA_IMAGES", DEFAULT_EXTRA_IMAGES)
    teacher_data = environment_path("TEACHER_DATA", DEFAULT_TEACHER_DATA)
    student_model = os.environ.get("STUDENT_MODEL", "yolov8n.pt")
    confidence = os.environ.get("TEACHER_CONFIDENCE", "0.70")
    epochs = int(os.environ.get("EPOCHS", "30"))

    pseudo_label_script = Path(__file__).with_name("pseudo_label_yolo11l.py")
    subprocess.run(
        [
            sys.executable,
            str(pseudo_label_script),
            "--teacher",
            teacher_model,
            "--source",
            str(extra_images),
            "--output",
            str(teacher_data),
            "--confidence",
            confidence,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    model = YOLO(student_model)
    model.train(
        data=str(teacher_data / "data.yaml"),
        epochs=epochs,
        imgsz=640,
        project=str(PROJECT_ROOT / "runs"),
        name="fitpose-teacher-student",
    )


if __name__ == "__main__":
    main()
