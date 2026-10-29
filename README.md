# FitPose AI — treinamento professor–aluno

Projeto de detecção de exercícios com YOLO. Este documento descreve somente o
fluxo de treinamento professor–aluno.

## Arquitetura do projeto

```text
imagens de entrada → API FastAPI → YoloDetector → modelo YOLO
                                      ↓
                         exercício + confiança + coordenadas
```

O dataset oficial fica em `data/fitness_normal/`. As imagens novas ficam em
`data/fitness_extra/images/`. O professor gera pseudo-rótulos em
`data/fitness_teacher/`, e o aluno aprende com esses exemplos antes de ser
usado pela API.

## Dataset adotado

Será utilizado o dataset [Fitness Normal](https://universe.roboflow.com/fitness-training/fitness_normal), do Roboflow Universe. Ele é do tipo **Object Detection**, possui 15 classes, aproximadamente 2.040 imagens e quatro versões publicadas. A licença indicada na página é CC BY 4.0; a versão escolhida e a data do download devem ser registradas no relatório final.

## Organização do projeto

- [Backlog do produto — Scrum FitPose IA](https://trello.com/invite/b/6adcc0b4711b032a5404431d/ATTIdc1957beef249391288eb2347b9343d2875AE613/🧠-scrum-fitpose-ia)
- [Quadro Kanban — Scrum FitPose IA](https://trello.com/invite/b/6adcc21bcf35cc1c29108cd2/ATTI7378846d928ed03fe9eaf33e6771dc8e5EB687B1/🧠-kanban-scrum-fitpose-ia)

Classes originais do dataset:

`Normal_BrenchPress`, `Normal_Deadlifts`, `Normal_Dips`, `Normal_DumbbellBicepCurls`, `Normal_JumpingJacks`, `Normal_LatPulldown`, `Normal_LegPress`, `Normal_Lunges`, `Normal_MountainClimbers`, `Normal_OverheadPress`, `Normal_Plunk`, `Normal_PullUps`, `Normal_PushUps`, `Normal_RowingMachineExercise` e `Normal_SquatsWithBarbell`.

Os nomes preservam as grafias originais (`Brench` e `Plunk`) para manter a
compatibilidade com o `data.yaml`. A apresentação pode usar nomes amigáveis.

## Estrutura Clean Architecture

```text
.
├── main.py                         # composition root da API
├── pyproject.toml                  # empacotamento e dependências
├── requirements.txt
├── mypy.ini                        # tipagem estática estrita
├── docs/
└── src/fitpose_ai/
    ├── domain/                     # entidades e interfaces (Protocol)
    ├── application/                # casos de uso
    ├── controller/                 # controllers HTTP/FastAPI
    ├── client/                     # cliente do modelo YOLO
    ├── config/                     # dataclasses de configuração
    └── composition.py              # injeção manual de dependências
```

O domínio não conhece FastAPI nem Ultralytics. O caso de uso depende da porta
`ExerciseDetector`, e `YoloDetector` encapsula o cliente YOLO. Essa separação
permite testar a regra de negócio com um detector falso, sem carregar o modelo.
As classes são imutáveis com `@dataclass(frozen=True)` e as funções possuem
anotações de tipo explícitas.

## Organização das camadas

`controller` representa a entrada HTTP, `client` encapsula a integração com o
modelo YOLO, `application` contém os casos de uso e `domain` contém as regras e
contratos.

## 1. Preparar o ambiente

```bash
cd /home/hebert/Documents/sctech/projeto-final
source .venv/bin/activate
```

O dataset oficial deve estar em `data/fitness_normal/`, contendo `data.yaml`,
`train/images`, `train/labels`, `valid/images` e `valid/labels`.

Coloque imagens novas, sem rótulos, em:

```text
data/fitness_extra/images/
```

## 2. Treinar o primeiro modelo

Este modelo será usado como professor e aprenderá os nomes dos exercícios a
partir do dataset oficial:

```bash
bash scripts/train_yolov8.sh
```

Ao terminar, o modelo estará em `models/best.pt`.

Guarde uma cópia para o benchmark:

```bash
cp models/best.pt models/best-base.pt
```

## 3. Gerar os rótulos das imagens novas

Use o modelo treinado do projeto como professor. Não use `yolo11l.pt`, pois ele
é genérico e pode identificar apenas `person`.

```bash
rm -rf data/fitness_teacher/images \
       data/fitness_teacher/labels \
       data/fitness_teacher/annotated

python scripts/pseudo_label_yolo.py \
  --teacher models/best.pt \
  --source data/fitness_extra/images \
  --output data/fitness_teacher \
  --confidence 0.50
```

O script cria:

```text
data/fitness_teacher/
├── images/        imagens aceitas pelo professor
├── labels/        coordenadas YOLO das caixas detectadas
├── annotated/     imagens com caixas e nomes desenhados
└── data.yaml      configuração do dataset do aluno
```

Confira as imagens em `annotated/`. Elas devem mostrar nomes de exercícios,
como `Normal_Plunk`, e não `person`.

## 4. Treinar o aluno final

```bash
yolo detect train \
  data=data/fitness_teacher/data.yaml \
  model=yolov8n.pt \
  epochs=30 \
  imgsz=640 \
  project=runs \
  name=fitpose-teacher-student
```

O melhor modelo ficará em:

```text
runs/fitpose-teacher-student/weights/best.pt
```

## 5. Usar o aluno na aplicação

```bash
cp runs/fitpose-teacher-student/weights/best.pt models/best.pt
```

## 6. Benchmark dos modelos

Compare os dois modelos usando a mesma validação oficial em
`data/fitness_normal/valid`. Não use as imagens pseudo-rotuladas para medir o
resultado final.

### Avaliar o modelo base

```bash
yolo detect val \
  model=models/best-base.pt \
  data=data/fitness_normal/data.yaml \
  split=val \
  project=runs/benchmark \
  name=base
```

### Avaliar o modelo professor–aluno

```bash
yolo detect val \
  model=models/best.pt \
  data=data/fitness_normal/data.yaml \
  split=val \
  project=runs/benchmark \
  name=teacher-student
```

Os resultados ficam em `runs/benchmark/base/` e
`runs/benchmark/teacher-student/`. Compare principalmente `mAP50-95`, `mAP50`,
precisão e recall. O modelo professor–aluno só deve ser considerado melhor se
apresentar ganho na mesma validação, sem reduzir significativamente precisão
ou recall.

## Fluxo resumido

```text
dataset oficial
      ↓
treinamento inicial → models/best.pt (professor)
      ↓
imagens novas → pseudo-rótulos e imagens anotadas
      ↓
treinamento final → best.pt do aluno
      ↓
copiar para models/best.pt
```

## Observações

- O arquivo `.txt` em `labels/` contém as coordenadas normalizadas das caixas.
- As imagens em `annotated/` servem para verificar os rótulos antes do treino.
- Se aparecer `person`, o comando foi executado com um modelo genérico em vez
  de `models/best.pt`.
- Não versionar imagens, pesos grandes ou a pasta `runs/` no Git.
