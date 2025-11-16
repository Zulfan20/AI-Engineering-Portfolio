# Portofolio AI Logo Generator

Ini adalah proyek portofolio *full-stack* yang menunjukkan implementasi *end-to-end* dari sistem AI generatif. Aplikasi ini memungkinkan pengguna untuk menghasilkan logo unik dari *prompt* teks, dengan menjalankan model AI (Stable Diffusion 1.5) secara lokal di GPU pengguna.

Proyek ini dipisahkan menjadi dua bagian:
1.  **Frontend (React):** Antarmuka pengguna (UI) modern yang dibuat dengan React, TypeScript, dan Vite.
2.  **Backend (Python):** Sebuah API asinkron berperforma tinggi yang dibuat dengan FastAPI untuk mengontrol GPU dan menjalankan model AI.

---

## Ì≥∏ Demo

*Sangat disarankan untuk Anda merekam layar Anda saat menggunakan aplikasi (dalam format GIF) dan meletakkannya di sini.*

`[Tempatkan GIF demo proyek Anda di sini. Ini sangat penting untuk portofolio!]`

---

## ‚ú® Fitur Utama

* **Generasi Teks-ke-Gambar:** Mengubah *prompt* teks (misal: "logo minimalis kepala robot") menjadi gambar logo.
* **Eksekusi GPU Lokal:** Menjalankan model AI Stable Diffusion 1.5 secara langsung di GPU NVIDIA (RTX 2050) pengguna melalui PyTorch dan CUDA, bukan bergantung pada API eksternal.
* **Arsitektur Full-Stack:** Memisahkan *frontend* dan *backend* untuk skalabilitas dan pemeliharaan yang lebih baik.
* **Backend API Cepat:** Menggunakan **FastAPI** untuk *backend* Python yang asinkron dan cepat.
* **Frontend Modern:** Menggunakan **React** dan **TypeScript** untuk UI yang interaktif dan aman secara tipe (*type-safe*).

---

## Ì≤ª Tumpukan Teknologi (Tech Stack)

| Kategori | Teknologi |
| :--- | :--- |
| **Frontend** | React, TypeScript, Vite, CSS |
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **AI / ML** | PyTorch, `diffusers` (Hugging Face), `accelerate`, CUDA |
| **Model AI** | `runwayml/stable-diffusion-v1.5` |
| **Manajemen** | Anaconda (Conda), NPM, Git |

---

## Ì∫Ä Cara Menjalankan Secara Lokal

Proyek ini terdiri dari dua server yang harus dijalankan secara bersamaan di dua terminal terpisah.

### Prasyarat

* **NVIDIA GPU** dengan dukungan CUDA.
* **Anaconda** (atau Miniconda).
* **Node.js** (v18+ direkomendasikan).

---

### 1. Ì∞ç Backend (Server Python FastAPI)

Server ini akan memuat model ke VRAM Anda dan menunggu *request* dari React.

1.  Buka **Anaconda Prompt**.
2.  Masuk ke folder *backend* proyek:
    ```bash
    cd C:\Users\zulfa\...\01-AI-LOGO-GENERATOR\ai-logo-backend
    ```
3.  Aktifkan *environment* Conda yang telah Anda buat:
    ```bash
    conda activate ai_backend
    ```
4.  Jalankan server API menggunakan Uvicorn:
    ```bash
    uvicorn main:app --reload
    ```
5.  **PENTING:** Biarkan terminal ini terbuka. Pertama kali dijalankan, server akan mengunduh model (beberapa GB) dan memuatnya ke GPU Anda. Ini mungkin butuh beberapa menit.
6.  Server Anda sekarang berjalan di `http://127.0.0.1:8000`.

---

### 2. ‚öõÔ∏è Frontend (Server React Vite)

Server ini akan menjalankan antarmuka pengguna (UI) di browser Anda.

1.  Buka **Terminal BARU** (Anda bisa menggunakan CMD, PowerShell, atau Anaconda Prompt kedua).
2.  Masuk ke folder *root* proyek (folder frontend):
    ```bash
    cd C:\Users\zulfa\...\01-AI-LOGO-GENERATOR\
    ```
3.  (Jika ini pertama kalinya) Instal semua dependensi Node.js:
    ```bash
    npm install
    ```
4.  Jalankan server *development* React:
    ```bash
    npm run dev
    ```
5.  **PENTING:** Biarkan terminal ini juga terbuka.

---

### 3. Buka Aplikasi

Buka browser Anda dan kunjungi alamat yang diberikan oleh server Vite (biasanya `http://localhost:5173`).

Anda sekarang dapat menggunakan aplikasi AI Logo Generator!
