#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="data/fitness_normal/data.yaml"

if [[ ! -f "$DATASET" ]]; then
    echo "Dataset não encontrado: $DATASET" >&2
    exit 1
fi

source .venv/bin/activate

yolo detect train \
    data="$DATASET" \
    model=yolov8n.pt \
    epochs=100 \
    patience=20 \
    imgsz=640 \
    project=runs \
    name=fitpose-yolov8

mkdir -p models

latest_best="$(
    find runs -type f -path '*/weights/best-v8.pt' -printf '%T@ %p\n' |
        sort -nr |
        head -n 1 |
        cut -d' ' -f2-
)"

if [[ -z "$latest_best" || ! -f "$latest_best" ]]; then
    echo "Nenhum best.pt foi encontrado em runs/." >&2
    exit 1
fi

cp "$latest_best" models/best-v8.pt

echo "Modelo selecionado: $latest_best"
echo "Modelo pronto em: $PROJECT_ROOT/models/best.pt"
