
# Portfolio Project 2: Advanced NLP - Transformer Sentiment Analysis
# Objective: Demonstrate Transfer Learning using a pre-trained Transformer model (BERT).

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import time
from sklearn.metrics import accuracy_score
import os

# --- 1. SETUP ---
# OTOMATIS mendeteksi GPU Anda. Ini adalah perubahan kuncinya.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
MODEL_PATH = 'NLP_Sentiment_Analysis/bert_sentiment_model'

print(f"Menggunakan device: {device}")
if torch.cuda.is_available():
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")

# --- 2. DATA LOADING & PREPROCESSING (IMDB Movie Reviews) ---
print("\n1. Memuat Dataset dan Tokenizer...")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_name = 'bert-base-uncased'
# Load 1000 samples for fast training
raw_datasets = load_dataset('imdb', split='train[:1000]') 
raw_datasets = raw_datasets.train_test_split(test_size=0.2, seed=42)
training_data = raw_datasets['train']
testing_data = raw_datasets['test']

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_train = training_data.map(tokenize_function, batched=True)
tokenized_test = testing_data.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.remove_columns(["text"]).rename_column("label", "labels").with_format("torch")
tokenized_test = tokenized_test.remove_columns(["text"]).rename_column("label", "labels").with_format("torch")

train_loader = DataLoader(tokenized_train, shuffle=True, batch_size=16)
test_loader = DataLoader(tokenized_test, batch_size=16)
print("Dataset berhasil dipersiapkan. Siap untuk Fine-Tuning.")


# --- 3. MODEL FINE-TUNING SETUP ---
print("\n2. Menginisialisasi Model Transformer (BERT)...")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) 
model.to(device) # Pindahkan model ke GPU

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) 

# --- 4. TRAINING LOOP (Fine-Tuning) ---
def train_loop(dataloader, model, optimizer, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        # Pindahkan data batch ke GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss. backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 5 == 0 and batch_idx > 0:
            pass # Suppress batch printing for cleaner output

    end_time = time.time()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} selesai dalam {(end_time - start_time):.2f} detik. Rata-rata Loss: {avg_loss:.4f}")

# --- 5. EVALUATION ---
def test_loop(dataloader, model):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Pindahkan data batch ke GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Pindahkan hasil kembali ke CPU untuk evaluasi (karena sklearn berjalan di CPU)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n--- Hasil Evaluasi ---")
    print(f"Akurasi Model: {accuracy:.4f}")
    return accuracy

# --- 6. RUN THE EXPERIMENT & SAVE ---
epochs = 2
print("\n--- Memulai Fine-Tuning BERT (2 Epochs) ---")
for t in range(epochs):
    train_loop(train_loader, model, optimizer, t + 1)
    
final_accuracy = test_loop(test_loader, model)

# Final step: Save the trained model to disk
# Ensure the directory exists before saving
os.makedirs(MODEL_PATH, exist_ok=True)
model.save_pretrained(MODEL_PATH) 
tokenizer.save_pretrained(MODEL_PATH) # Also save the tokenizer
print(f"\nModel BERT berhasil disimpan ke folder: {MODEL_PATH}")
print("Project NLP Selesai! Siap untuk didokumentasikan di GitHub.")