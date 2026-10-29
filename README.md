# FitPose AI

Microsserviço FastAPI para detecção de posturas de musculação com YOLO — Projeto avaliativo, Módulo 2.

## Dataset adotado

Será utilizado o dataset [Fitness Normal](https://universe.roboflow.com/fitness-training/fitness_normal), do Roboflow Universe. Ele é do tipo **Object Detection**, possui 15 classes, aproximadamente 2.040 imagens e quatro versões publicadas. A licença indicada na página é CC BY 4.0; a versão escolhida e a data do download devem ser registradas no relatório final.

## Organização do projeto

- [Backlog do produto — Scrum FitPose IA](https://trello.com/invite/b/6adcc0b4711b032a5404431d/ATTIdc1957beef249391288eb2347b9343d2875AE613/🧠-scrum-fitpose-ia)
- [Quadro Kanban — Scrum FitPose IA](https://trello.com/invite/b/6adcc21bcf35cc1c29108cd2/ATTI7378846d928ed03fe9eaf33e6771dc8e5EB687B1/🧠-kanban-scrum-fitpose-ia)

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

## Scripts disponíveis

A pasta `scripts/` reúne os comandos de automação do treinamento e da
pseudo-rotulagem. Eles servem para padronizar a execução do projeto sem repetir
comandos longos no terminal:

- `scripts/train_yolov8.sh`: treina o modelo base `yolov8n.pt` com o dataset
  principal `data/fitness_normal`, identifica o `best.pt` mais recente em
  `runs/` e copia para `models/best.pt`.
- `scripts/pseudo_label_yolo.py`: usa um modelo professor (padrão `yolo11l.pt`)
  para analisar imagens novas em `data/fitness_extra/images` e gerar rótulos YOLO
  em `data/fitness_teacher/` com confiança configurável (`--confidence`). As
  imagens ficam em `images/`, as coordenadas em `labels/` e as cópias anotadas
  com as caixas desenhadas ficam em `annotated/`.
- `scripts/train_teacher_student.sh`: executa a sequência completa de
  pseudo-rotulagem + treinamento do aluno (`yolov8n.pt`), preservando a última
  comparação com o dataset original em `data/fitness_normal`.

Exemplos de uso:

```bash
bash scripts/train_yolov8.sh
python scripts/pseudo_label_yolo.py --teacher yolo11l.pt --source data/fitness_extra/images --output data/fitness_teacher --confidence 0.70
bash scripts/train_teacher_student.sh
```

Se o ambiente estiver com a virtualenv ativa, esses comandos rodam sem a
necessidade de repetir o fluxo inteiro manualmente.

## Guia completo de execução

Este projeto possui dois modos independentes de treinamento. Escolha apenas
um deles conforme o objetivo:

### Modo 1 — Treinamento tradicional

Treina diretamente o `yolov8n.pt` com os rótulos oficiais do dataset
`fitness_normal`. É o caminho mais simples e serve como modelo base para
comparar resultados.

Fluxo:

```text
dataset original → yolov8n → models/best.pt → API
```

Execute as etapas de preparação do ambiente e download do dataset abaixo e,
depois, rode:

```bash
bash scripts/train_yolov8.sh
cp runs/fitpose-yolov8/weights/best.pt models/best.pt
uvicorn main:app --reload
```

### Modo 2 — Professor–aluno

Usa um YOLO maior (`yolo11l.pt`) para gerar pseudo-rótulos de imagens extras
e depois treina novamente o `yolov8n.pt`. Para medir o ganho corretamente,
primeiro treinamos um aluno base, depois executamos o professor e, por fim,
treinamos o aluno final com os novos dados. O professor não recebe o aluno
base como entrada; ele é usado para produzir os rótulos adicionais.

Fluxo:

```text
dataset original → yolov8n aluno base
imagens extras → yolo11l professor → pseudo-rótulos
dataset pseudo-rotulado → yolov8n aluno final → API
```

Neste modo, execute as etapas de preparação e dataset abaixo, coloque as
imagens extras em `data/fitness_extra/images` e siga a sequência específica
da seção `Professor–aluno automatizado`.

### 1. Preparar o ambiente

Requisitos: Python 3.11+, Git e pelo menos 8 GB de RAM. GPU NVIDIA com CUDA é
recomendada, mas não obrigatória; sem GPU, o treinamento usa CPU e demora mais.

```bash
cd caminho/para/fitpose-ai
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

### 3. Treinar um modelo base (opcional, para comparação)

Antes do fluxo professor–aluno, você pode treinar um YOLOv8n somente com o
dataset original. Isso cria a linha de base para saber se o professor
realmente trouxe melhoria. Este passo não é o modelo final obrigatório:

```bash
./scripts/train_yolov8.sh
```

Ele treina com YOLOv8n, 50 épocas e imagens 640x640. Ao terminar, localiza o
`best.pt` da execução mais recente, copia o melhor peso para `models/best.pt` e
mantém os resultados em `runs/`. O log deve mostrar `CUDA:0` quando a GPU
NVIDIA for detectada.

### 4. Fluxo recomendado: professor e depois aluno

No fluxo final, a ordem correta é:

```text
imagens extras → professor yolo11l → pseudo-rótulos → aluno yolov8n → best.pt
```

Siga a seção [Professor–aluno automatizado](#professoraluno-automatizado)
abaixo. Só depois de concluir o treinamento do aluno copie o `best.pt` para
`models/best.pt` e inicie a API.

### 5. Iniciar a API

```bash
uvicorn main:app --reload
```

Abra `http://127.0.0.1:8000/docs`.

### 6. Publicar a API na internet com ngrok

Para permitir que outras pessoas testem a API remotamente, mantenha a API
rodando em um terminal e, em outro, inicie um túnel do ngrok para a porta
`8000`.

1. Instale o [ngrok](https://ngrok.com/download), crie uma conta gratuita e
   configure o seu token de autenticação conforme as instruções do site.
2. Com o ambiente virtual ativado e a API em execução, rode:

```bash
ngrok http 8000
```

O ngrok exibirá uma URL pública semelhante a
`https://abcd-1234.ngrok-free.app`. Compartilhe essa URL para que outras
pessoas acessem a documentação em `/docs` ou façam requisições para o endpoint
`/detect`, por exemplo:

```bash
curl -X POST https://abcd-1234.ngrok-free.app/detect \
  -F "file=@caminho/para/exercicio.jpg"
```

A URL muda quando o túnel é reiniciado no plano gratuito. O túnel também deixa
de funcionar quando o comando do ngrok é encerrado. Não publique modelos,
tokens ou outros arquivos sensíveis e encerre o túnel após os testes.

### 7. Fazer uma inferência

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
  aparece `CUDA:0 (SUA PLACA DE VIDEO)`.
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

Após validar o treinamento, copiar `runs/fitpose-yolov8/weights/best.pt` para `models/best.pt`.

---

### Professor–aluno automatizado

#### Execução passo a passo

Use exatamente esta sequência para executar e comparar os dois alunos:

**1. Clonar e entrar no projeto**

```bash
git clone <URL_DO_REPOSITORIO>
cd projeto-final
```

**2. Criar o ambiente e instalar as dependências**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**3. Baixar o dataset original** (opcional, pois já está dentro do projeto)

Baixe a versão 4 do
[fitness_normal](https://universe.roboflow.com/fitness-training/fitness_normal)
no formato YOLOv8, extraia o ZIP em `data/fitness_normal` e confirme que
`data/fitness_normal/data.yaml` existe.

**4. Colocar as imagens extras**

Crie a pasta e coloque nela as imagens do dataset
[gym-exercise-detection](https://universe.roboflow.com/roheans-workspace/gym-exercise-detection):

```bash
mkdir -p data/fitness_extra/images
```

Copie somente arquivos `.jpg`, `.jpeg` ou `.png` para
`data/fitness_extra/images`. Não copie os `labels` do ZIP; o professor fará
os rótulos compatíveis com este projeto.

**5. Treinar o aluno base**

Este primeiro treinamento usa somente os rótulos oficiais e serve de
referência para medir se a etapa professor–aluno trouxe melhoria.

```bash
bash scripts/train_yolov8.sh
```

O modelo base fica em `models/best.pt`. Preserve esse arquivo ou renomeie-o
para `models/baseline-best.pt` antes de continuar.

**6. Executar o professor**

```bash
python scripts/pseudo_label_yolo.py \
  --teacher yolo11l.pt \
  --source data/fitness_extra/images \
  --output data/fitness_teacher \
  --confidence 0.70
```

Esse comando pode baixar `yolo11l.pt` automaticamente. Ao terminar, confirme
que foram criados `data/fitness_teacher/images`,
`data/fitness_teacher/labels` e `data/fitness_teacher/data.yaml`.

**7. Treinar o aluno final**

```bash
yolo detect train \
  data=data/fitness_teacher/data.yaml \
  model=yolov8n.pt \
  epochs=30 \
  imgsz=640 \
  project=runs \
  name=fitpose-teacher-student
```

O aluno é o `yolov8n.pt`, escolhido por ser mais leve para execução local.

**8. Instalar o aluno final na API**

```bash
mkdir -p models
cp runs/fitpose-teacher-student/weights/best.pt models/best.pt
uvicorn main:app --reload
```

Abra `http://127.0.0.1:8000/docs` e teste `POST /detect`. O resultado final
do treinamento fica em `runs/fitpose-teacher-student/weights/best.pt`.

Se quiser executar automaticamente apenas as etapas de professor e aluno
final, use o script abaixo depois de preparar as imagens. O aluno base deve
ser treinado separadamente no passo 5:

```bash
bash scripts/train_teacher_student.sh
```

#### Motivação da abordagem

Como este projeto está sendo construído do zero, inicialmente temos poucos
dados específicos e queremos que o modelo final seja leve o suficiente para
rodar localmente. Por isso usamos dois modelos com funções diferentes:

- o modelo professor (`yolo11l.pt`) é maior e tem mais capacidade para
  encontrar pessoas e exercícios nas imagens;
- o modelo aluno (`yolov8n.pt`) é menor, mais rápido e adequado para a API
  local.

O professor analisa imagens adicionais e gera pseudo-rótulos no formato YOLO.
Depois, o aluno aprende com esses exemplos. Essa técnica é conhecida como
aprendizado professor–aluno (*teacher–student*) ou pseudo-rotulagem.

Essa abordagem é vantajosa para um projeto iniciado do zero porque reduz o
trabalho de anotação manual, aproveita um modelo visual já pré-treinado e
permite aumentar o conjunto de treinamento com dados do ambiente real da
aplicação. Também separa qualidade de treinamento e custo de execução: o
professor pode ser pesado durante a preparação dos dados, enquanto o modelo
leve permanece na API.

O professor não substitui completamente dados corretos. Ele pode transferir
erros para o aluno, especialmente quando uma imagem não pertence a uma das
15 classes do projeto. Por isso usamos um limite de confiança, mantemos os
rótulos originais intactos e fazemos a avaliação final em um conjunto de
validação separado. O ganho só é considerado válido se o modelo treinado com
pseudo-rótulos superar o modelo base no mesmo conjunto de teste.

#### Reprodução local completa

Depois de clonar o projeto, prepare o ambiente:

```bash
git clone <URL_DO_REPOSITORIO>
cd projeto-final
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Também é necessário ter o dataset original em
`data/fitness_normal`, contendo `data.yaml`, `train/images`, `train/labels`,
`valid/images` e `valid/labels`. O dataset original é o
[fitness_normal no Roboflow](https://universe.roboflow.com/fitness-training/fitness_normal),
versão 4, licença CC BY 4.0.

Para repetir a etapa do professor, coloque imagens adicionais em
`data/fitness_extra/images` e execute:

```bash
python scripts/pseudo_label_yolo.py \
  --teacher yolo11l.pt \
  --source data/fitness_extra/images \
  --output data/fitness_teacher \
  --confidence 0.70
```

O professor pode baixar `yolo11l.pt` automaticamente na primeira execução.
Ele cria imagens e rótulos compatíveis em `data/fitness_teacher`. Os rótulos
originais do dataset adicional não são usados diretamente.

Para treinar o aluno local (`yolov8n`), execute:

```bash
yolo detect train \
  data=data/fitness_teacher/data.yaml \
  model=yolov8n.pt \
  epochs=30 \
  imgsz=640 \
  project=runs \
  name=fitpose-teacher-student
```

O resultado final será salvo em:

```text
runs/fitpose-teacher-student/weights/best.pt
```

Para usar o aluno na API, copie o melhor peso para o caminho esperado pela
aplicação:

```bash
mkdir -p models
cp runs/fitpose-teacher-student/weights/best.pt models/best.pt
uvicorn main:app --reload
```

Depois, abra `http://127.0.0.1:8000/docs` e teste `POST /detect`. Para
verificar se o treinamento terminou, confirme a existência de
`best.pt`, `results.csv` e `results.png` dentro de
`runs/fitpose-teacher-student`.

O script que executa professor e aluno em sequência também está disponível:

```bash
bash scripts/train_teacher_student.sh
```

Ele usa `yolo11l.pt` como professor, `yolov8n.pt` como aluno e confiança
`0.70` por padrão. Para alterar esses valores:

```bash
TEACHER_MODEL=yolo11x.pt TEACHER_CONFIDENCE=0.80 \
  bash scripts/train_teacher_student.sh
```

Para avaliar o resultado, use sempre a validação original em
`data/fitness_normal/data.yaml`. Não use as imagens pseudo-rotuladas para
medir a qualidade final, pois isso produziria uma avaliação otimista.

Foi adicionada uma etapa automatizada de aprendizado professor–aluno. O
dataset adicional utilizado foi o
[gym-exercise-detection](https://universe.roboflow.com/roheans-workspace/gym-exercise-detection),
do Roboflow Universe, com licença **CC BY 4.0**. Foram extraídas 553 imagens
para `data/fitness_extra/images`. O dataset contém exemplos de `arm_circle`,
`plank`, `pullup`, `pushup` e `squat`.

Os rótulos originais do download não são utilizados diretamente. O modelo
professor gera novos rótulos usando os nomes e IDs das 15 classes do projeto,
para que o conjunto seja compatível com o YOLO local. A origem, licença e
data de obtenção do dataset devem ser mantidas no relatório do projeto.

Para gerar os pseudo-rótulos sem alterar os rótulos originais do dataset
principal, execute:

```bash
./scripts/train_teacher_student.sh
```

Por padrão, `yolo11l.pt` atua como professor e `yolov8n.pt` como aluno. Os
pesos do professor podem ser alterados com `TEACHER_MODEL`, e o limite de
confiança com `TEACHER_CONFIDENCE=0.80`. O conjunto de validação original não
é usado para pseudo-rotulagem. Imagens pseudo-rotuladas devem ser novas; usar
as mesmas imagens do treino original não adiciona informação e pode apenas
reproduzir erros do professor.

O modelo treinado fica em:

```text
runs/fitpose-teacher-student/weights/best.pt
```

Para colocá-lo em uso na API:

```bash
mkdir -p models
cp runs/fitpose-teacher-student/weights/best.pt models/best.pt
uvicorn main:app --reload
```

A comparação final deve ser feita usando o conjunto de validação original em
`data/fitness_normal`, e não os pseudo-rótulos gerados pelo professor.

| Modelo | Precisão | Recall | mAP50 | Inferência | Decisão |
|---|---:|---:|---:|---:|---|
| YOLOv8n | a medir | a medir | a medir | a medir | candidato inicial |
| YOLOv11n | a medir | a medir | a medir | a medir | opcional |
