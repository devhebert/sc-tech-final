from csv import DictWriter
from pathlib import Path

from ultralytics import YOLO

modelos = {
    "yolov8l": "runs/fitpose-yolov8l/weights/best.pt",
    "yolov8n": "runs/fitpose-yolov8n/weights/best.pt",
}

resultados = []

for nome, caminho in modelos.items():
    print(f"\nAvaliando o modelo: {nome}\n")

    metricas = YOLO(caminho).val(
        data="data/fitness_normal/data.yaml",
        split="val",
        project="runs/benchmark",
        name=nome,
        exist_ok=True,
    )

    resultados.append({
        "modelo": nome,
        "precision": metricas.box.mp,
        "recall": metricas.box.mr,
        "mAP50": metricas.box.map50,
        "mAP50-95": metricas.box.map,
    })

Path("runs/benchmark").mkdir(parents=True, exist_ok=True)

with open("runs/benchmark/metrics.csv", "w", newline="", encoding="utf-8") as arquivo:
    escritor = DictWriter(arquivo, fieldnames=resultados[0].keys())
    escritor.writeheader()
    escritor.writerows(resultados)