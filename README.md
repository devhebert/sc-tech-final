# FitPose AI

Microsserviço FastAPI para detecção de posturas de musculação com YOLO — Projeto avaliativo, Módulo 2, Semana 14.

## Dataset adotado

Será utilizado o dataset [Fitness Normal](https://universe.roboflow.com/fitness-training/fitness_normal), do Roboflow Universe. Ele é do tipo **Object Detection**, possui 15 classes, aproximadamente 2.040 imagens e quatro versões publicadas. A licença indicada na página é CC BY 4.0; a versão escolhida e a data do download devem ser registradas no relatório final.

Classes originais do dataset:

`Normal_BrenchPress`, `Normal_Deadlifts`, `Normal_Dips`, `Normal_DumbbellBicepCurls`, `Normal_JumpingJacks`, `Normal_LatPulldown`, `Normal_LegPress`, `Normal_Lunges`, `Normal_MountainClimbers`, `Normal_OverheadPress`, `Normal_Plunk`, `Normal_PullUps`, `Normal_PushUps`, `Normal_RowingMachineExercise` e `Normal_SquatsWithBarbell`.

Os nomes apresentam algumas grafias originais (`Brench`, `Plunk`). Eles serão preservados no modelo para manter compatibilidade com o `data.yaml`; a apresentação pode usar nomes amigáveis.


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
`ExerciseDetector`, e `YoloDetector` é encapsulado pelo `YoloClient`. Essa separação
permite testar a regra de negócio com um detector falso, sem carregar o modelo.
As classes são imutáveis com `@dataclass(frozen=True)` e as funções possuem
anotações de tipo explícitas.

## Organização das camadas

`controller` representa a entrada HTTP, `client` encapsula a integração com o
modelo YOLO, `application` contém os casos de uso e `domain` contém as regras e
contratos.

## Guia completo de execução

### 1. Preparar o ambiente

Requisitos: Python 3.11+, Git e pelo menos 8 GB de RAM. GPU NVIDIA com CUDA é
recomendada, mas não obrigatória; sem GPU, o treinamento usa CPU e demora mais.

```bash
cd /home/hebert/Documents/sctech/projeto-final
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

No Windows, use `.venv\\Scripts\\activate` no lugar do comando `source`.

### 2. Baixar e extrair o dataset

Abra o [Fitness Normal no Roboflow Universe](https://universe.roboflow.com/fitness-training/fitness_normal), selecione a versão 4, escolha **Baixar conjunto de dados**, formato **YOLOv8**, e baixe o ZIP.

```bash
mkdir -p data/fitness_normal
unzip ~/Downloads/fitness_normal.v4i.yolov8.zip -d data/fitness_normal
```

Confirme que existe `data/fitness_normal/data.yaml`. O dataset contém as
pastas `train/images`, `train/labels`, `valid/images` e `valid/labels`.

### 3. Treinar o modelo

Execute o script pronto:

```bash
./scripts/train_yolov8.sh
```

Ele treina com YOLOv8n, 50 épocas e imagens 640x640. Ao terminar, localiza o
`best.pt` da execução mais recente, copia o melhor peso para `models/best.pt` e
mantém os resultados em `runs/`. O log deve mostrar `CUDA:0` quando a GPU
NVIDIA for detectada.

### 4. Iniciar a API

```bash
uvicorn main:app --reload
```

Abra `http://127.0.0.1:8000/docs`.

### 5. Fazer uma inferência

No Swagger, abra `POST /detect`, clique em **Try it out**, selecione uma imagem
de exercício e clique em **Execute**. Pela linha de comando:

```bash
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@caminho/para/exercicio.jpg"
```

Teste a disponibilidade com `curl http://127.0.0.1:8000/health`. Sem
`models/best.pt`, `/detect` responde HTTP 503; o Swagger apenas executa
inferência e não treina o modelo.

## Verificação de tipagem

O retorno da API preserva `class_name`, nome original do dataset, e adiciona
`label`, nome amigável traduzido para português.

```bash
mypy src main.py
```

## Solução de problemas

- **Dataset não encontrado:** confirme `data/fitness_normal/data.yaml` e as
  pastas `train/images` e `valid/images`.
- **Modelo não encontrado:** execute `./scripts/train_yolov8.sh` e confirme
  `ls -lh models/best.pt`.
- **CPU em vez de GPU:** o treinamento continua funcionando; confirme no log se
  aparece `CUDA:0 (NVIDIA GeForce RTX 4050)`.
- **Unsupported image type:** envie uma imagem válida (JPG, JPEG ou PNG). A API
  converte o upload para RGB antes de executar o YOLO.

## Dataset e benchmark
No Roboflow, gerar a versão escolhida e exportar em formato **YOLOv8**. O arquivo baixado deve conter `data.yaml`, imagens e rótulos YOLO. Não versionar imagens ou pesos grandes no GitHub; registrar o link, versão, licença e instruções de download.

## Treinamento

No Google Colab ou em uma máquina com GPU:

```bash
pip install ultralytics
yolo detect train data=/caminho/fitness-normal/data.yaml model=yolov8n.pt epochs=50 imgsz=640 project=runs name=fitpose-yolov8
```

Após validar o treinamento, copiar `runs/fitpose-yolov8/weights/best.pt` para `models/best.pt`. O modelo não é incluído no repositório por ser um arquivo grande.

| Modelo | Precisão | Recall | mAP50 | Inferência | Decisão |
|---|---:|---:|---:|---:|---|
| YOLOv8n | a medir | a medir | a medir | a medir | candidato inicial |
| YOLOv11n | a medir | a medir | a medir | a medir | opcional |

## Kanban e entrega
Colunas sugeridas: Backlog, A Fazer, Em andamento, Revisão/Testes e Concluído. Cartões: escolher versão do dataset, exportar YOLO, treinar YOLOv8, validar métricas, implementar API, testar Swagger/Postman, gravar vídeo e publicar.

## Entrega final

Conferir: repositório GitHub, README, documentação funcional e técnica, link do dataset, versão e licença, métricas reais do treinamento, modelo funcional, quadro Trello público, vídeo com acesso de leitura e os links submetidos no AVA.
