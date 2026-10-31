# FitPose AI

Projeto de detecção de exercícios com YOLO e API FastAPI. O projeto treina e
compara duas variantes do YOLOv8 usando o mesmo dataset e a mesma divisão de
validação:

- YOLOv8n;
- YOLOv8l.

## Organização do projeto

- [Backlog do produto — Scrum FitPose IA](https://trello.com/invite/b/6adcc0b4711b032a5404431d/ATTIdc1957beef249391288eb2347b9343d2875AE613/🧠-scrum-fitpose-ia)
- [Quadro Kanban — Scrum FitPose IA](https://trello.com/invite/b/6adcc21bcf35cc1c29108cd2/ATTI7378846d928ed03fe9eaf33e6771dc8e5EB687B1/🧠-kanban-scrum-fitpose-ia)

## Arquitetura

```text
imagem → API FastAPI → YoloDetector → modelo YOLO
                                      ↓
                         exercício + confiança + coordenadas
```

O domínio não conhece FastAPI nem Ultralytics. O caso de uso depende da porta
`ExerciseDetector`, enquanto `YoloDetector` encapsula a integração com o
modelo YOLO.

## Dataset

O dataset utilizado é o [Fitness Normal](https://universe.roboflow.com/fitness-training/fitness_normal),
do Roboflow Universe. Ele é um dataset de **Object Detection** com 15 classes,
licença CC BY 4.0 e 4 versões publicadas. A versão escolhida foi a *v4 2025-03-11* que contém 5706 imagens, 
divididas em:

Treino (96%): 5499 imagens   
Validaçao (4%): 207 imagens

O dataset pode ser encontrado em `data/fitness_normal/` com a seguinte estrutura:

```text
data/fitness_normal/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

As classes definidas em `data/fitness_normal/data.yaml` são:

```text
Normal_BrenchPress
Normal_Deadlifts
Normal_Dips
Normal_DumbbellBicepCurls
Normal_JumpingJacks
Normal_LatPulldown
Normal_LegPress
Normal_Lunges
Normal_MountainClimbers
Normal_OverheadPress
Normal_Plunk
Normal_PullUps
Normal_PushUps
Normal_RowingMachineExercise
Normal_SquatsWithBarbell
```

Os nomes preservam as grafias originais do dataset, como `Brench` e `Plunk`,
para manter a compatibilidade com o arquivo `data.yaml`.

## Estrutura do projeto (Clean Architecture)

```text
.
├── main.py
├── pyproject.toml
├── requirements.txt
├── mypy.ini
├── data/fitness_normal/
├── models/
├── runs/
├── scripts/
└── src/fitpose_ai/
    ├── domain/       # entidades e interfaces
    ├── application/  # casos de uso
    ├── controller/   # endpoints FastAPI
    ├── client/       # integração com YOLO
    ├── config/       # configurações da aplicação
    └── composition.py
```

## Organização das camadas

`controller` representa a entrada HTTP, `client` encapsula a integração com o
modelo YOLO, `application` contém os casos de uso e `domain` contém as regras e
contratos.

## 1. Preparação do ambiente

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

OBS: os comandos devem ser executados na raíz do projeto.

## 2. Treinamento dos modelos

O script `scripts/train_yolov8_variants.py` treina, por padrão, as variantes
`yolov8n.pt` e `yolov8l.pt` usando:

- 100 épocas;
- `imgsz=512`;
- `batch=4`;
- `patience=20`;
- dataset `data/fitness_normal/data.yaml`.

Execute:

```powershell
python scripts\train_yolov8_variants.py
```

O script também pode ser configurado por variáveis de ambiente, por exemplo:

```powershell
$env:EPOCHS="50"
$env:IMGSZ="640"
$env:MODELS="yolov8n.pt yolov8l.pt"
python scripts\train_yolov8_variants.py
```

Ao final, os melhores pesos ficam em:

```text
runs/fitpose-yolov8n/weights/best.pt
runs/fitpose-yolov8l/weights/best.pt
```

Os arquivos de cada treinamento, incluindo `results.csv`, ficam na respectiva
pasta dentro de `runs/`.

O script `scripts/train_yolov8_variants.sh` é a alternativa para ambientes com
Bash. Execute-o com:

```bash
bash scripts/train_yolov8_variants.sh
```

## 3. Avaliação durante o treinamento

Durante o treinamento, o YOLO avalia o modelo ao final de cada época usando o
conjunto de validação definido em `data.yaml`. Essa etapa acompanha a evolução
do modelo e ajuda a selecionar o `best.pt`.

Os principais arquivos gerados são:

```text
results.csv
results.png
confusion_matrix.png
F1_curve.png
PR_curve.png
P_curve.png
R_curve.png
```

Essa avaliação é diferente do benchmark final: o benchmark executa uma nova
avaliação padronizada dos pesos finais dos dois modelos.

## 4. Executação do benchmark

O benchmark principal é executado por:

```powershell
python scripts\benchmark.py
```

O script:

1. avalia `runs/fitpose-yolov8l/weights/best.pt`;
2. avalia `runs/fitpose-yolov8n/weights/best.pt`;
3. imprime no terminal qual modelo está sendo avaliado e as métricas correspondentes;
4. salva as métricas consolidadas em `runs/benchmark/metrics.csv`;
5. salva TP, FP e FN por classe em `runs/benchmark/metrics_by_class.csv`.

As imagens e os gráficos de cada avaliação ficam em:

```text
runs/benchmark/yolov8l/
runs/benchmark/yolov8n/
```

Os dois modelos devem usar o mesmo dataset, a mesma divisão `val` e a mesma
configuração de validação. O script reutiliza os arquivos `best.pt` já
existentes e não executa um novo treinamento.

Além de precisão, recall e mAP, o arquivo `metrics.csv` registra:

- `TP_total`: detecções corretas;
- `FP_total`: detecções sem correspondência correta com um rótulo;
- `FN_total`: objetos anotados que não foram detectados;
- `acertos_brutos`: total de verdadeiros positivos no conjunto de validação;
- `latencia_inferencia_ms_img`: tempo médio de inferência por imagem;
- `FPS`: quadros por segundo, calculado como `1000 / ms_por_imagem`.

O arquivo `metrics_by_class.csv` detalha `TP`, `FP` e `FN` para cada classe.
Os valores são derivados da matriz de confusão produzida durante a validação.
Assim, recall alto representa menos exercícios perdidos, enquanto precisão alta
representa menos classificações incorretas. A latência e o FPS medidos referem-se
à execução no ambiente em que o benchmark foi executado.

### Avaliação individual

Se não for necessário gerar o CSV consolidado, cada modelo pode ser avaliado
individualmente:

```powershell
yolo detect val \
  model=runs\fitpose-yolov8l\weights\best.pt \
  data=data\fitness_normal\data.yaml \
  split=val \
  project=runs\benchmark \
  name=yolov8l
```

```powershell
yolo detect val \
  model=runs\fitpose-yolov8n\weights\best.pt \
  data=data\fitness_normal\data.yaml \
  split=val \
  project=runs\benchmark \
  name=yolov8n
```

Esses comandos exibem as métricas no terminal e geram os gráficos e imagens,
mas não geram necessariamente um arquivo CSV consolidado.

### Resultado do benchmark

Os resultados registrados em `runs/benchmark/metrics.csv` foram:

| Métrica | YOLOv8l | YOLOv8n |
|---|---:|---:|
| Precisão | 0,9690 | 0,8294 |
| Recall | 1,0000 | 0,8728 |
| mAP50 | 0,9950 | 0,9169 |
| mAP50-95 | 0,8831 | 0,6164 |
| TP total | 207 | 201 |
| FP total | 0 | 24 |
| FN total | 0 | 6 |
| Acertos brutos | 207 | 201 |
| Latência de inferência | 855,79 ms/imagem | 125,43 ms/imagem |
| FPS | 1,17 | 7,97 |

Significado das métricas:

- **TP (True Positive ou verdadeiro positivo):** detecção correta, quando a
  previsão corresponde a um objeto realmente presente e à sua classe.
- **FP (False Positive ou falso positivo):** detecção incorreta, quando o
  modelo prevê um objeto que não corresponde corretamente a um rótulo real.
- **FN (False Negative ou falso negativo):** objeto presente no rótulo que não
  foi detectado pelo modelo.
- **FPS (Frames Per Second ou quadros por segundo):** quantidade estimada de
  imagens processadas por segundo, calculada por `1000 / latência_ms_por_imagem`.

O YOLOv8l foi escolhido para a API por apresentar o melhor desempenho de
detecção: teve 207 verdadeiros positivos, não produziu falsos positivos nem
falsos negativos e alcançou recall de 1,0000. Também obteve precisão de 0,9690,
mAP50 de 0,9950 e mAP50-95 de 0,8831. Essa última é a métrica mais rigorosa,
pois considera diferentes limiares de IoU e avalia melhor a qualidade do
posicionamento das caixas.

O YOLOv8n é mais rápido, com 7,97 FPS contra 1,17 FPS do YOLOv8l, mas apresenta
menor qualidade: 24 falsos positivos, 6 falsos negativos e 201 acertos brutos.
Assim, o YOLOv8n é aproximadamente 6,8 vezes mais rápido, enquanto o YOLOv8l
oferece maior confiabilidade e precisão para o objetivo do sistema. Os valores
de velocidade foram medidos durante a validação no ambiente do benchmark e
podem variar conforme o hardware.

## 5. Execução da API

O modelo escolhido deve ser copiado para `models/best.pt`, que é o caminho
padrão utilizado pela API:

```powershell
Copy-Item runs\fitpose-yolov8l\weights\best.pt models\best.pt
```

Para iniciar a API, execute:

```powershell
python -m uvicorn main:app --reload
```

Por meio da variável `MODEL_PATH`, é possível especificar outro caminho, se necessário:

```powershell
$env:MODEL_PATH="outro\caminho\modelo.pt"
python -m uvicorn main:app --reload
```

Também é possível configurar o limiar de confiança:

```powershell
$env:CONFIDENCE_THRESHOLD="0.50"
python -m uvicorn main:app --reload
```

Endpoints disponíveis:

- `GET /health`: verifica o status da API e informa o modelo carregado;
- `POST /detect`: recebe uma imagem e retorna as detecções encontradas.

Exemplo de resposta do endpoint `POST /detect`:

```json
{
  "status": "success",
  "detections": [
    {
      "class_name": "Normal_OverheadPress",
      "label": "Desenvolvimento de ombros",
      "confidence": 0.3724,
      "box": {
        "x_min": 20,
        "y_min": 0,
        "x_max": 402,
        "y_max": 360
      }
    }
  ]
}
```

Após iniciar a aplicação, a documentação interativa fica disponível em
`http://127.0.0.1:8000/docs`.
