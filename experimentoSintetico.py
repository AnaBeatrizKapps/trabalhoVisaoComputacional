import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import random
import time

# Configurações Gerais
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DATASET_SIZE = 10000  # Número de imagens sintéticas para o dataset
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Caminho para a fonte
IMG_SIZE = (128, 32)

# Função para gerar imagens sintéticas
def generate_synthetic_image(text, font_path=FONT_PATH, img_size=IMG_SIZE):
    font = ImageFont.truetype(font_path, 20)
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), text, font=font, fill='black')
    return img

# Dataset Sintético
class SyntheticOCRDataset(Dataset):
    def __init__(self, size, vocab, font_path=FONT_PATH, img_size=IMG_SIZE):
        self.size = size
        self.vocab = vocab
        self.font_path = font_path
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        text_length = random.randint(5, 15)  # Comprimento aleatório do texto
        text = ''.join(random.choices(self.vocab, k=text_length))
        image = generate_synthetic_image(text, self.font_path, self.img_size)
        return {"image": image, "text": text}

# Vocabulário sintético
VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")

# Gerar o dataset sintético
synthetic_dataset = SyntheticOCRDataset(DATASET_SIZE, VOCAB)

# Dividir em treino e teste
train_size = int(0.8 * DATASET_SIZE)
test_size = DATASET_SIZE - train_size
train_data, test_data = torch.utils.data.random_split(synthetic_dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# Modelo TrOCR
model_trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(DEVICE)
tokenizer_trocr = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")

# Modelo DTrOCR
model_dtrocr = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
tokenizer_dtrocr = AutoTokenizer.from_pretrained("gpt2")

# Otimizadores
optimizer_trocr = torch.optim.AdamW(model_trocr.parameters(), lr=LEARNING_RATE)
optimizer_dtrocr = torch.optim.AdamW(model_dtrocr.parameters(), lr=LEARNING_RATE)

# Função de Pré-processamento
def preprocess_data(batch, processor):
    pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values
    input_ids = processor.tokenizer(batch["text"], return_tensors="pt", padding="max_length", truncation=True).input_ids
    return {"pixel_values": pixel_values, "input_ids": input_ids}

# Função de Treinamento
def train_model(model, data_loader, processor, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data_loader:
            pixel_values = []
            input_ids = []
            for item in batch:
                processed = preprocess_data(item, processor)
                pixel_values.append(processed["pixel_values"])
                input_ids.append(processed["input_ids"])

            pixel_values = torch.cat(pixel_values).to(DEVICE)
            input_ids = torch.cat(input_ids).to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader)}")

# Função de Avaliação
def evaluate_model(model, tokenizer, data_loader, processor):
    model.eval()
    total_cer = 0
    total_wer = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = []
            input_ids = []
            for item in batch:
                processed = preprocess_data(item, processor)
                pixel_values.append(processed["pixel_values"])
                input_ids.append(processed["input_ids"])

            pixel_values = torch.cat(pixel_values).to(DEVICE)
            input_ids = torch.cat(input_ids).to(DEVICE)

            outputs = model.generate(pixel_values)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            for pred, label in zip(preds, labels):
                cer = sum(1 for a, b in zip(pred, label) if a != b) / max(len(label), 1)
                wer = len(set(pred.split()) ^ set(label.split())) / max(len(label.split()), 1)

                total_cer += cer
                total_wer += wer

    inference_time = time.time() - start_time
    cer_avg = total_cer / len(data_loader.dataset)
    wer_avg = total_wer / len(data_loader.dataset)
    return cer_avg, wer_avg, inference_time

# Treinamento TrOCR
print("Treinando TrOCR com dados sintéticos...")
train_model(model_trocr, train_loader, AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten"), optimizer_trocr, EPOCHS)

# Avaliação TrOCR
print("Avaliando TrOCR com dados sintéticos...")
cer_trocr, wer_trocr, time_trocr = evaluate_model(model_trocr, tokenizer_trocr, test_loader, AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten"))
print(f"TrOCR - CER: {cer_trocr:.2f}, WER: {wer_trocr:.2f}, Tempo de Inferência: {time_trocr:.2f}s")

# Treinamento DTrOCR
print("Treinando DTrOCR com dados sintéticos...")
train_model(model_dtrocr, train_loader, AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten"), optimizer_dtrocr, EPOCHS)

# Avaliação DTrOCR
print("Avaliando DTrOCR com dados sintéticos...")
cer_dtrocr, wer_dtrocr, time_dtrocr = evaluate_model(model_dtrocr, tokenizer_dtrocr, test_loader, AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten"))
print(f"DTrOCR - CER: {cer_dtrocr:.2f}, WER: {wer_dtrocr:.2f}, Tempo de Inferência: {time_dtrocr:.2f}s")
