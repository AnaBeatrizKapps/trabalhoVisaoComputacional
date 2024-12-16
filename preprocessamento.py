import os
import pandas as pd

def parse_ascii(ascii_path):
    """
    Processa os arquivos ASCII para obter as transcrições e IDs de imagem.
    """
    data = []
    with open(ascii_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("a"):  # Cada linha relevante começa com 'a'
                parts = line.strip().split(" ")
                image_id = parts[0]  # Exemplo: a01-000u-00
                transcription = " ".join(parts[8:])  # A transcrição começa após o 8º elemento
                data.append({"image_id": image_id, "text": transcription})
    return pd.DataFrame(data)

# Caminhos para os arquivos
ascii_dir = "datasets/iam/ascii/"
lines_dir = "datasets/iam/lines/"

# Parse dos arquivos ASCII
all_data = []
for ascii_file in os.listdir(ascii_dir):
    if ascii_file.endswith(".txt"):
        ascii_path = os.path.join(ascii_dir, ascii_file)
        all_data.append(parse_ascii(ascii_path))

# Combina todos os dados
dataset = pd.concat(all_data, ignore_index=True)

# Adiciona caminhos completos para as imagens
def build_image_path(image_id):
    """
    Constrói o caminho correto para as imagens no diretório `lines`.
    Exemplo: a01-000u-00 -> lines/a01/a01-000u/a01-000u-00.png
    """
    parts = image_id.split("-")
    dir1 = parts[0]  # Exemplo: a01
    dir2 = f"{parts[0]}-{parts[1]}"  # Exemplo: a01-000u
    filename = f"{image_id}.png"  # Exemplo: a01-000u-00.png
    return os.path.join(lines_dir, dir1, dir2, filename)

# Aplica a correção aos caminhos das imagens
dataset["image_path"] = dataset["image_id"].apply(build_image_path)

# Salva o dataset corrigido
dataset.to_csv("iam_dataset_corrected.csv", index=False)
