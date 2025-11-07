# Project 1: GPU-Accelerated Transfer Learning for Image Classification

This project is a core component of my AI Engineering portfolio, demonstrating the successful implementation of a modern computer vision pipeline. The objective is to fine-tune a pre-trained **ResNet18** model for the **CIFAR-10** image classification task, leveraging **NVIDIA GPU acceleration (RTX 2050)** for efficient training.

## 🎯 Objective

The primary goal is to showcase proficiency in advanced deep learning techniques, specifically:
* **Transfer Learning:** Implementing a state-of-the-art technique by adapting a large, pre-trained model (ResNet18) to a new, specific task.
* **GPU Acceleration:** Successfully configuring a PyTorch environment (`torch-gpu`) to utilize NVIDIA CUDA for massive training speedups.
* **Full CV Pipeline:** Managing the complete end-to-end workflow, including data loading (`DataLoader`), complex image transformations (`transforms`), model definition, training, and evaluation.

## 🛠️ Technologies Used

* **Python 3.x**
* **PyTorch & Torchvision:** For building, training, and loading the deep learning model.
* **NVIDIA CUDA:** For GPU acceleration (running on an NVIDIA GeForce RTX 2050).

## 📈 Methodology

1.  **Data Loading:** The CIFAR-10 dataset (60,000 32x32 images across 10 classes) is loaded using `torchvision.datasets`.
2.  **Preprocessing:** Images are transformed to meet the requirements of the ResNet model: resized to 224x224 and normalized using ImageNet's standard mean and deviation.
3.  **Model Definition:**
    * A pre-trained **ResNet18** model is loaded from `torchvision.models`.
    * All parameters of the model are "frozen" (`requires_grad = False`) to retain the learned ImageNet features.
    * The final classification layer (`model.fc`) is **replaced** with a new, untrained `nn.Linear` layer customized for CIFAR-10's 10 classes.
4.  **Training (Fine-Tuning):**
    * The model and data are moved to the `cuda` (GPU) device.
    * The model is trained for 3 epochs, but **only the weights of the new final layer** are optimized.
5.  **Evaluation:** The model's final accuracy is calculated on the unseen test set.

## 📊 Results

This demonstrates the power of transfer learning, successfully training a complex model on a full dataset.

* **Hardware:** NVIDIA GeForce RTX 2050 (via `torch-gpu` Conda env)
* **Device:** `cuda`
* **Average Epoch Time:** `~252.3 seconds`
    * *(Note: Performance bottleneck identified in the CPU-bound `transforms.Resize(224)` operation, not GPU computation.)*
* **Final Test Accuracy:** `79.5%`

Achieving nearly 80% accuracy in just 3 epochs highlights the effectiveness of fine-tuning a pre-trained model, a task made feasible by the GPU.

## 🚀 How to Run

1.  Ensure you have a Python environment with PyTorch and Torchvision installed (e.g., the `torch-gpu` conda environment).
2.  Navigate to the project directory.
3.  Run the script:
    ```bash
    python 08_CV_Transfer_Learning.py
    ```