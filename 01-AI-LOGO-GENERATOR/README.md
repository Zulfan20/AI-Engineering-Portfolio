# AI Logo Generator Portfolio Project

This is a full-stack project demonstrating an end-to-end implementation of a generative AI system. The application allows users to generate unique logos from text prompts by running the AI model (Stable Diffusion 1.5) locally on the user's GPU.

The project is architected into two main parts:
1.  **Frontend (React):** A modern user interface built with React, TypeScript, and Vite.
2.  **Backend (Python):** A high-performance, asynchronous API built with FastAPI to control the GPU and run the AI model.

---

## 📸 Demo

*It is highly recommended to record a GIF of your application in use and place it here. This is crucial for a portfolio.*

`[Place your project demo GIF here!]`

---

## ✨ Key Features

* **Text-to-Image Generation:** Transforms text prompts (e.g., "a minimalist logo of a robot head") into images.
* **Local GPU Execution:** Runs the Stable Diffusion 1.5 model directly on the user's NVIDIA GPU (RTX 2050) via PyTorch and CUDA, rather than relying on an external API.
* **Full-Stack Architecture:** Decouples the frontend and backend for better scalability and maintenance.
* **Fast API Backend:** Uses **FastAPI** for an asynchronous, high-speed Python backend.
* **Modern Frontend:** Uses **React** and **TypeScript** for an interactive, type-safe UI.

---

## 💻 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Frontend** | React, TypeScript, Vite, CSS |
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **AI / ML** | PyTorch, `diffusers` (Hugging Face), `accelerate`, CUDA |
| **AI Model** | `runwayml/stable-diffusion-v1.5` |
| **Management** | Anaconda (Conda), NPM, Git |

---

## 🚀 How to Run Locally

This project consists of two separate servers that must be run simultaneously in two separate terminals.

### Prerequisites

* An **NVIDIA GPU** with CUDA support.
* **Anaconda** (or Miniconda).
* **Node.js** (v18+ recommended).

---

### 1. 🐍 Backend (Python FastAPI Server)

This server will load the AI model into your VRAM and listen for requests from the React app.

1.  Open **Anaconda Prompt**.
2.  Navigate to the project's backend folder:
    ```bash
    cd C:\Users\zulfa\...\01-AI-LOGO-GENERATOR\ai-logo-backend
    ```
3.  Activate the Conda environment you created:
    ```bash
    conda activate ai_backend
    ```
4.  Run the API server using Uvicorn:
    ```bash
    uvicorn main:app --reload
    ```
5.  **IMPORTANT:** Leave this terminal open. On the first run, the server will download the model (several GBs) and load it onto your GPU. This may take a few minutes.
6.  Your server is now running at `http://127.0.0.1:8000`.

---

### 2. ⚛️ Frontend (React Vite Server)

This server will run the user interface in your browser.

1.  Open a **NEW Terminal** (you can use CMD, PowerShell, or a second Anaconda Prompt).
2.  Navigate to the project's root folder (the frontend folder):
    ```bash
    cd C:\Users\zulfa\...\01-AI-LOGO-GENERATOR\
    ```
3.  (If this is your first time) Install all Node.js dependencies:
    ```bash
    npm install
    ```
4.  Run the React development server:
    ```bash
    npm run dev
    ```
5.  **IMPORTANT:** Leave this terminal open as well.

---

### 3. Open the Application

Open your browser and navigate to the address provided by the Vite server (usually `http://localhost:5173`).

You can now use the AI Logo Generator application!