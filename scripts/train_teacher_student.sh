#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

TEACHER_MODEL="${TEACHER_MODEL:-yolo11l.pt}"
EXTRA_IMAGES="${EXTRA_IMAGES:-data/fitness_extra/images}"
TEACHER_DATA="${TEACHER_DATA:-data/fitness_teacher}"

# As imagens de entrada devem ser novas ou pertencer ao ambiente real de uso.
# Reutilizar exatamente o treino original não acrescenta informação e pode
# apenas transferir os erros do professor para o aluno.
python scripts/pseudo_label_yolo.py \
  --teacher "$TEACHER_MODEL" \
  --source "$EXTRA_IMAGES" \
  --output "$TEACHER_DATA" \
  --confidence "${TEACHER_CONFIDENCE:-0.70}"

yolo detect train \
  data="$TEACHER_DATA/data.yaml" \
  model="${STUDENT_MODEL:-yolov8n.pt}" \
  epochs="${EPOCHS:-30}" \
  imgsz=640 \
  project=runs \
  name=fitpose-teacher-student

# Compare o resultado deste treinamento com o modelo original usando a
# validação de data/fitness_normal/data.yaml; não use os pseudo-rótulos para
# medir a qualidade final.
