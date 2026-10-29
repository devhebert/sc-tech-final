#!/usr/bin/env python3
"""Gera rótulos YOLO automaticamente usando um modelo professor."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="yolo11l.pt", help="Peso do professor")
    parser.add_argument("--source", type=Path, required=True, help="Pasta com imagens sem rótulo")
    parser.add_argument("--output", type=Path, default=Path("data/fitness_teacher"))
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Pasta de imagens não encontrada: {args.source}")

    image_files = sorted(
        path for path in args.source.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_files:
        raise SystemExit("Nenhuma imagem encontrada na pasta de origem.")

    # O dataset pseudo-rotulado fica separado do dataset original. Assim,
    # podemos comparar os experimentos e desfazer esta etapa sem perder
    # nenhum rótulo revisado manualmente.
    images_dir = args.output / "images"
    labels_dir = args.output / "labels"
    annotated_dir = args.output / "annotated"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    # O professor é um detector maior/especializado. Ele gera caixas para
    # imagens novas; não usamos a validação original como entrada aqui.
    model = YOLO(args.teacher)
    accepted = 0
    for image_path in image_files:
        result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.confidence, verbose=False)[0]
        # O parâmetro conf já elimina previsões abaixo do limite configurado.
        # Imagens sem nenhuma detecção confiável não entram no treino do aluno.
        if result.boxes is None or len(result.boxes) == 0:
            continue

        destination_image = images_dir / image_path.name
        destination_label = labels_dir / f"{image_path.stem}.txt"
        destination_annotated = annotated_dir / image_path.name
        shutil.copy2(image_path, destination_image)
        # Guarda uma cópia visual com as caixas e coordenadas desenhadas pelo
        # Ultralytics para facilitar a conferência dos pseudo-rótulos.
        plotted_image = result.plot(labels=True, conf=True, boxes=True)
        from PIL import Image
        Image.fromarray(plotted_image[:, :, ::-1]).save(destination_annotated)
        height, width = result.orig_shape
        with destination_label.open("w", encoding="utf-8") as label_file:
            for box, class_id, confidence in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                x1, y1, x2, y2 = box.tolist()
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                label_file.write(
                    f"{int(class_id.item())} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}\n"
                )
        accepted += 1

    # Este YAML serve apenas para treinar com os pseudo-rótulos. A avaliação
    # final deve continuar usando o data.yaml original e sua validação intacta.
    data_yaml = args.output / "data.yaml"
    # O caminho absoluto evita que o Ultralytics resolva `images` dentro do
    # diretório global de datasets quando o treinamento é iniciado por outro
    # diretório ou por uma configuração externa.
    data_yaml.write_text(
        f"path: {args.output.resolve()}\ntrain: images\nval: images\n"
        "nc: 15\n"
        "names: ['Normal_BrenchPress', 'Normal_Deadlifts', 'Normal_Dips', "
        "'Normal_DumbbellBicepCurls', 'Normal_JumpingJacks', 'Normal_LatPulldown', "
        "'Normal_LegPress', 'Normal_Lunges', 'Normal_MountainClimbers', "
        "'Normal_OverheadPress', 'Normal_Plunk', 'Normal_PullUps', 'Normal_PushUps', "
        "'Normal_RowingMachineExercise', 'Normal_SquatsWithBarbell']\n",
        encoding="utf-8",
    )
    print(f"Pseudo-rótulos aceitos: {accepted}/{len(image_files)}")
    print(f"Dataset gerado em: {args.output}")


if __name__ == "__main__":
    main()
