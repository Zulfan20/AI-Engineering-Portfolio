# Portfolio Project 1: GPU-Accelerated Computer Vision
# Objective: Demonstrate Transfer Learning using a pre-trained ResNet18 model on the CIFAR-10 dataset.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os

# --- 1. SETUP & DEVICE CHECK ---
# Ini akan secara otomatis mendeteksi dan memilih RTX 2050 Anda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Menggunakan Device: {device} ---")
if torch.cuda.is_available():
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")

# --- 2. DATA PREPARATION (CIFAR-10 Dataset) ---
print("\n1. Memuat & Mempersiapkan Data CIFAR-10...")

# Model ResNet membutuhkan input 224x224 dan normalisasi khusus
transform = transforms.Compose([
    transforms.Resize(224), # Ubah ukuran gambar kecil (32x32) menjadi 224x224
    transforms.ToTensor(),
    # Ini adalah nilai Mean dan Std Dev standar untuk dataset ImageNet
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# Unduh dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Buat DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Data berhasil dimuat.")

# --- 3. MODEL DEFINITION (TRANSFER LEARNING) ---
print("\n2. Menginisialisasi Model ResNet18 (Transfer Learning)...")

# 3.1 Muat model ResNet18 yang sudah dilatih di ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 3.2 Bekukan semua parameter di model dasar
# Kita tidak ingin melatih ulang seluruh model, hanya lapisan terakhir
for param in model.parameters():
    param.requires_grad = False

# 3.3 Ganti lapisan klasifikasi terakhir (disebut 'fc' di ResNet)
# Dapatkan jumlah fitur input dari lapisan 'fc'
num_ftrs = model.fc.in_features

# Buat lapisan Linear baru untuk 10 kelas CIFAR-10
model.fc = nn.Linear(num_ftrs, 10) 

# 3.4 Pindahkan seluruh model ke GPU
model = model.to(device)
print("Model ResNet18 berhasil dimodifikasi dan dipindah ke GPU.")

# --- 4. TRAINING SETUP ---
loss_fn = nn.CrossEntropyLoss()
# Beri tahu optimizer untuk HANYA memperbarui parameter dari lapisan 'fc' yang baru
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# --- 5. TRAINING LOOP (Fine-Tuning) ---
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    model.train() # Set model ke mode training
    total_loss = 0
    start_time = time.time()
    
    for batch, (X, y) in enumerate(dataloader):
        # Pindahkan data batch ke GPU
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch % 100 == 0: # Cetak progres setiap 100 batch
            print(f"  [Batch {batch}/{len(dataloader)}] Loss: {loss.item():.4f}")

    end_time = time.time()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} selesai dalam {(end_time - start_time):.2f} detik. Rata-rata Loss: {avg_loss:.4f}")

# --- 6. EVALUATION ---
def test_loop(dataloader, model, loss_fn):
    model.eval() # Set model ke mode evaluasi (penting!)
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad(): # Tidak perlu menghitung gradien saat evaluasi
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"\n--- Hasil Evaluasi ---")
    print(f"Akurasi Model: {(100*correct):>0.1f}% \n")

# --- 7. RUN THE EXPERIMENT ---
epochs = 3 # Kita hanya perlu 3 epoch untuk hasil yang baik dengan transfer learning
print("\n--- Memulai Fine-Tuning ResNet18 (3 Epochs) ---")
for t in range(epochs):
    print(f"Epoch {t+1}")
    train_loop(train_loader, model, loss_fn, optimizer, t+1)
    
test_loop(test_loader, model, loss_fn)
print("Proyek Computer Vision Selesai!")