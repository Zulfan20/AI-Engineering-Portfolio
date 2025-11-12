# AI Engineer & Data Scientist Portfolio - [YOUR NAME]

Welcome to my professional portfolio. I am a [Your Title, e.g., Computer Science Student / AI Engineer] with a passion for building efficient and impactful AI solutions. In this repository, you will find 5 end-to-end projects that demonstrate my skills in *Machine Learning*, *Deep Learning* (CV & NLP), *Generative AI*, and *Deployment*.

**Email:** `[zulfanisious20@gmail.com]` | **LinkedIn:** `[www.linkedin.com/in/muhammad-zulfan-abidin-b4427b212]`

---

## Featured Projects

Here is a summary of the 5 end-to-end projects I have completed:

### 1. [Project 1] Customer Churn Prediction Web App (End-to-End)

* **Project Folder:** `Telco_Churn_Predictor/`
* **Objective:** To build an interactive web application to predict customer churn using a classic *machine learning* model.
* **Skills Demonstrated:**
    * **Traditional ML:** Trained a **Random Forest Classifier** model using Scikit-learn with 98.5% accuracy.
    * **Deployment:** Used **Streamlit** to deploy the model (`.joblib`) as a functional and interactive UI.
    * **Full Lifecycle:** Demonstrated the entire workflow: from raw data -> training -> model saving -> web application.

### 2. [Project 2] Text Sentiment Analysis (Advanced NLP)

* **Project Folder:** `NLP_Sentiment_Analysis/`
* **Objective:** To use *Transfer Learning* to *fine-tune* a Transformer (BERT) model for a text classification task.
* **Skills Demonstrated:**
    * **Transformer Architecture:** Successfully implemented and trained a **BERT** model (`bert-base-uncased`).
    * **NLP Pipeline:** Handled *tokenization*, *data loading* (Hugging Face `datasets`), and *fine-tuning* on a movie review dataset.
    * **Model Analysis:** Identified *overfitting* (100% accuracy) on a small sample dataset (1,000) and understood the need for a larger dataset for generalization.

### 3. [Project 3] Image Classification (GPU-Accelerated CV)

* **Project Folder:** `Computer_Vision_Classifier/`
* **Objective:** To use *Transfer Learning* with **ResNet18** to classify images, accelerated by an **NVIDIA RTX 2050**.
* **Skills Demonstrated:**
    * **GPU Acceleration:** Successfully configured the PyTorch (`torch-gpu`) environment to use **CUDA**, which significantly accelerated training.
    * **Computer Vision:** Implemented a modern CV pipeline, including *image transformation* (resizing 32x32 -> 224x224) and normalization.
    * **Results:** Achieved **79.5% accuracy** on the CIFAR-10 dataset in just 3 *epochs*, proving the effectiveness of *Transfer Learning*.

### 4. [Project 4] Sketch-to-Reality (Generative AI & GPU)

* **Project Folder:** `Sketch_to_Reality/`
* **Objective:** To build an end-to-end web application that converts simple user sketches into photorealistic images using generative AI.
* **Skills Demonstrated:**
    * **Generative AI:** Implemented a **Stable Diffusion ControlNet** pipeline (using `diffusers`) to generate images based on both text prompts and sketch inputs.
    * **GPU Optimization:** Configured the environment to run on a local **NVIDIA RTX 2050**, including installing `torch` with CUDA support and using `enable_sequential_cpu_offload` to manage VRAM.
    * **Full-Stack:** Built a **Flask/Python** backend to serve the AI model and a **JavaScript/HTML Canvas** frontend for the user's drawing interface.

### 5. [Project 5] AI Logo Generator (Gemini API)

* **Project Folder:** `AI_Logo_Generator/` (or your folder name)
* **Objective:** To create a simple, self-contained web app to generate logos using Google's latest `imagen` model via the Gemini API.
* **Skills Demonstrated:**
    * **API Integration:** Successfully integrated with a major cloud AI service (Google Gemini API) to perform image generation.
    * **Frontend Development:** Built a clean, responsive UI in a single **HTML/JavaScript** file, including asynchronous `fetch` calls.
    * **Error Handling:** Implemented API rate-limit handling using an exponential backoff strategy to ensure a robust user experience.

---

## ️ Technical Environment

* **Frameworks:** PyTorch, Scikit-learn, Transformers (Hugging Face), Diffusers (Hugging Face), Streamlit, Flask, Pandas, NumPy
* **Cloud / APIs:** Google Gemini API (Imagen)
* **Hardware:** NVIDIA GeForce RTX 2050 (CUDA Acceleration)
* **Tools:** VS Code, Git, Anaconda (Conda Environments), JavaScript (Canvas)