#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${DATASET:-data/fitness_normal/data.yaml}"
#MODELS="${MODELS:-yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt}"
MODELS="${MODELS:-yolov8n.pt yolov8l.pt}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-2}"
RUN_PREFIX="${RUN_PREFIX:-fitpose}"

if [[ ! -f "$DATASET" ]]; then
    echo "Dataset não encontrado: $DATASET" >&2
    exit 1
fi

source .venv/bin/activate
mkdir -p models

for model in $MODELS; do
    model_name="${model%.pt}"
    run_name="${RUN_PREFIX}-${model_name}"
    output_name="${model_name}.pt"

    echo "Treinando: $model -> runs/$run_name"
    yolo detect train \
        data="$DATASET" \
        model="$model" \
        epochs="$EPOCHS" \
        patience="$PATIENCE" \
        imgsz="$IMGSZ" \
        batch="$BATCH" \
        workers="$WORKERS" \
        cache=False \
        amp=True \
        project=runs \
        name="$run_name"

    best_path="runs/${run_name}/weights/best.pt"
    if [[ ! -f "$best_path" ]]; then
        echo "Checkpoint não encontrado: $best_path" >&2
        exit 1
    fi

    cp "$best_path" "models/$output_name"
    echo "Modelo pronto em: $PROJECT_ROOT/models/$output_name"
done
