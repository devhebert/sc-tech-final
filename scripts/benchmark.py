from csv import DictWriter
from pathlib import Path

from ultralytics import YOLO

modelos = {
    "yolov8l": "runs/fitpose-yolov8l/weights/best.pt",
    "yolov8n": "runs/fitpose-yolov8n/weights/best.pt",
}

resultados = []
resultados_por_classe = []

for nome, caminho in modelos.items():
    print(f"\nAvaliando o modelo: {nome}\n")

    metricas = YOLO(caminho).val(
        data="data/fitness_normal/data.yaml",
        split="val",
        project="runs/benchmark",
        name=nome,
        exist_ok=True,
        plots=True,
    )

    matriz = metricas.confusion_matrix.matrix
    quantidade_classes = len(metricas.names)
    nomes_classes = [
        metricas.names[index] for index in range(quantidade_classes)
    ]
    verdadeiros_positivos = matriz.diagonal()[:quantidade_classes]
    falsos_positivos = (
        matriz[:quantidade_classes, :].sum(axis=1) - verdadeiros_positivos
    )
    falsos_negativos = (
        matriz[:, :quantidade_classes].sum(axis=0) - verdadeiros_positivos
    )

    for classe, tp, fp, fn in zip(
        nomes_classes, verdadeiros_positivos, falsos_positivos, falsos_negativos
    ):
        resultados_por_classe.append({
            "modelo": nome,
            "classe": classe,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
        })

    tp_total = int(verdadeiros_positivos.sum())
    fp_total = int(falsos_positivos.sum())
    fn_total = int(falsos_negativos.sum())
    inferencia_ms = float(metricas.speed["inference"])

    resultados.append({
        "modelo": nome,
        "precision": metricas.box.mp,
        "recall": metricas.box.mr,
        "mAP50": metricas.box.map50,
        "mAP50-95": metricas.box.map,
        "TP_total": tp_total,
        "FP_total": fp_total,
        "FN_total": fn_total,
        "acertos_brutos": tp_total,
        "latencia_inferencia_ms_img": inferencia_ms,
        "FPS": 1000 / inferencia_ms,
    })

diretorio_benchmark = Path("runs/benchmark")
diretorio_benchmark.mkdir(parents=True, exist_ok=True)

with (diretorio_benchmark / "metrics.csv").open(
    "w", newline="", encoding="utf-8"
) as arquivo:
    escritor = DictWriter(arquivo, fieldnames=resultados[0].keys())
    escritor.writeheader()
    escritor.writerows(resultados)

with (diretorio_benchmark / "metrics_by_class.csv").open(
    "w", newline="", encoding="utf-8"
) as arquivo:
    escritor = DictWriter(arquivo, fieldnames=resultados_por_classe[0].keys())
    escritor.writeheader()
    escritor.writerows(resultados_por_classe)