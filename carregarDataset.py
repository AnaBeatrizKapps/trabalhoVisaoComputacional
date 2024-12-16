import pandas as pd
from PIL import Image

# Carregar dataset processado
data = pd.read_csv("iam_dataset_corrected.csv")

# Função para carregar dados
def load_image_and_text(row):
    try:
        # Tenta carregar a imagem e a transcrição
        image = Image.open(row["image_path"]).convert("RGB")
        text = row["text"]
        return {"image": image, "text": text}
    except FileNotFoundError:
        print(f"Imagem não encontrada: {row['image_path']}")
        return None

# Exemplo de carregamento
sample = data.iloc[0]
image_and_text = load_image_and_text(sample)
if image_and_text:
    print(f"Texto: {image_and_text['text']}")
    image_and_text["image"].show()
