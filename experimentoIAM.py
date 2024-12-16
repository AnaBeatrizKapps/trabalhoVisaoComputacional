import os
from datasets import Dataset
import torch
from torch.utils.data import DataLoader, random_split
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import time

# Configurações Gerais
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Função para carregar e organizar o IAM Handwriting Database localmente
def load_iam_dataset(base_path):
    data = {"image": [], "text": []}
    transcription_path = os.path.join(base_path, "ascii")
    image_path = os.path.join(base_path, "lines")
    
    for root, _, files in os.walk(transcription_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("#") or not line.strip():
                            continue
                        parts = line.split(" ", 1)
                        if len(parts) < 2:
                            continue
                        image_file = parts[0].replace("-", "/") + ".png"
                        text = parts[1].strip()
                        full_image_path = os.path.join(image_path, image_file)
                        if os.path.exists(full_image_path):
                            data["image"].append(full_image_path)
                            data["text"].append(text)
    return Dataset.from_dict(data)

# Caminho para os dados locais
base_path = "datasets/iam"
iam_dataset = load_iam_dataset(base_path)

# Dividir em treino e teste
train_test_split = iam_dataset.train_test_split(test_size=0.2)
train_data = train_test_split["train"]
test_data = train_test_split["test"]

# Função de pré-processamento
def preprocess_data(batch):
    processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    pixel_values = processor(images=batch["image"], return_tensors="pt").pixel_values
    input_ids = processor.tokenizer(batch["text"], return_tensors="pt", padding="max_length", truncation=True).input_ids
    return {"pixel_values": pixel_values, "input_ids": input_ids}

# Pré-processar dados
train_data = train_data.map(preprocess_data, batched=True)
test_data = test_data.map(preprocess_data, batched=True)
train_data.set_format(type="torch", columns=["pixel_values", "input_ids"])
test_data.set_format(type="torch", columns=["pixel_values", "input_ids"])

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

# Função de Treinamento
def train_model(model, data_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            
            outputs = model(pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader)}")

# Função de Avaliação
def evaluate_model(model, tokenizer, data_loader):
    model.eval()
    total_cer = 0
    total_wer = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)

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

# Treinamento
print("Treinando TrOCR...")
train_model(model_trocr, train_loader, optimizer_trocr, EPOCHS)

print("Treinando DTrOCR...")
train_model(model_dtrocr, train_loader, optimizer_dtrocr, EPOCHS)

# Avaliação
print("Avaliando TrOCR...")
cer_trocr, wer_trocr, time_trocr = evaluate_model(model_trocr, tokenizer_trocr, test_loader)
print(f"TrOCR - CER: {cer_trocr:.2f}, WER: {wer_trocr:.2f}, Tempo de Inferência: {time_trocr:.2f}s")

print("Avaliando DTrOCR...")
cer_dtrocr, wer_dtrocr, time_dtrocr = evaluate_model(model_dtrocr, tokenizer_dtrocr, test_loader)
print(f"DTrOCR - CER: {cer_dtrocr:.2f}, WER: {wer_dtrocr:.2f}, Tempo de Inferência: {time_dtrocr:.2f}s")
