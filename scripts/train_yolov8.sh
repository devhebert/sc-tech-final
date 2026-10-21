#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "data/fitness_normal/data.yaml" ]]; then
  echo "Dataset não encontrado: data/fitness_normal/data.yaml" >&2
  exit 1
fi

source .venv/bin/activate
yolo detect train \
  data=data/fitness_normal/data.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  project=runs \
  name=fitpose-yolov8

mkdir -p models
latest_best="$(find runs -type f -path '*/weights/best.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$latest_best" || ! -f "$latest_best" ]]; then
  echo "Nenhum best.pt foi encontrado em runs/." >&2
  exit 1
fi
cp "$latest_best" models/best.pt
echo "Modelo selecionado: $latest_best"
echo "Modelo pronto em: $PROJECT_ROOT/models/best.pt"
