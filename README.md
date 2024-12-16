# trabalho Visão Computacional
## Análise Comparativa dos Métodos TrOCR e DTrOCR para Reconhecimento de Texto Manuscrito

Este projeto realiza experimentos utilizando os modelos TrOCR e DTrOCR para o reconhecimento de caracteres.

## Passos para Execução

### 1. **Configuração do Ambiente**
Certifique-se de ter um ambiente Python configurado. Recomenda-se o uso de um ambiente virtual.

#### Criação do Ambiente Virtual:
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/MacOS
myenv\Scripts\activate   # Windows
```

### 2. **Instalação das Dependências**
Instale as bibliotecas necessárias executando os comandos abaixo:

```bash
pip install --upgrade datasets
pip install torch torchvision torchaudio
pip install transformers
```

### 3. **Estrutura do Projeto**
Certifique-se de que os arquivos do projeto estão organizados da seguinte forma:

```
project/
├── experimentoSintetico.py  # Script principal para os experimentos
├── experimentoIAM.py    # Script principal para os experimentos com o dataset IAM
├── README.md                   # Instruções (este arquivo)
└── fonts/                      # Diretório contendo as fontes
    └── DejaVuSans.ttf          # Fonte para geração de imagens sintéticas
└── datasets/                      # Diretório contendo as fontes
    └── iam          # Diretório contendo os arquivos extraídos
```

### 4. **Execução do Script**
Execute o script principal para iniciar os experimentos:

```bash
python experimentoSintetico.py
```

O script irá:
- Gerar um dataset sintético com imagens de texto.
- Treinar os modelos TrOCR e DTrOCR com os dados gerados.
- Avaliar os modelos em termos de CER (Character Error Rate), WER (Word Error Rate) e tempo de inferência.

### 5. **Experimento com o Dataset IAM**
Para realizar os experimentos com o IAM Handwriting Database:

#### 5.1 **Baixar os Arquivos Necessários**
Acesse o site oficial do IAM Handwriting Database: [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). Registre-se e baixe os seguintes arquivos:

- `ascii.tgz`: Contém informações de transcrição e metadados.
- `lines.tgz`: Contém as imagens das linhas de texto manuscrito.

#### 5.2 **Organizar os Arquivos**
Após extrair os arquivos, organize-os na seguinte estrutura:

```
datasets/
└── iam/
    ├── ascii/
    │   ├── *.txt  # Arquivos de informações meta
    ├── lines/
    │   ├── a01/
    │   │   ├── a01-000/
    │   │   │   ├── a01-000-00.png
```

#### 5.3 **Execução do Script**
Execute o script `experimentoIAM.py` para iniciar os experimentos com o dataset IAM:

```bash
python experimentoIAM.py
```
O script irá:
- Processar as transcrições do IAM Handwriting Database.
- Vincular as imagens às suas transcrições correspondentes.
- Treinar os modelos TrOCR e DTrOCR com os dados reais do IAM.
- Avaliar os modelos em termos de CER, WER e tempo de inferência.

### 6. **Configurações Personalizáveis**
Nos scripts, você pode ajustar os seguintes parâmetros:
- **DATASET_SIZE** (para dados sintéticos): Número de amostras sintéticas a serem geradas.
- **FONT_PATH** (para dados sintéticos): Caminho para a fonte a ser usada na geração de imagens.
- **BATCH_SIZE**: Tamanho do lote para treinamento.
- **EPOCHS**: Número de épocas para treinamento.

### 7. **Resultados**
Após a execução, os resultados do treinamento e avaliação serão exibidos no console. As métricas incluem:
- **CER (Character Error Rate)**: Taxa de erro de caracteres.
- **WER (Word Error Rate)**: Taxa de erro de palavras.
- **Tempo de Inferência**: Tempo médio para processar cada amostra.
