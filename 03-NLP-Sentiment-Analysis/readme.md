Project 1: GPU-Accelerated Transfer Learning for Image Classification

This project is a core component of my AI Engineering portfolio, demonstrating the successful implementation of a modern computer vision pipeline. The objective is to fine-tune a pre-trained ResNet18 model for the CIFAR-10 image classification task, leveraging NVIDIA GPU acceleration (RTX 2050) for efficient training.

🎯 Objective

The primary goal is to showcase proficiency in advanced deep learning techniques, specifically:

Transfer Learning: Implementing a state-of-the-art technique by adapting a large, pre-trained model (ResNet18) to a new, specific task.

GPU Acceleration: Successfully configuring a PyTorch environment (torch-gpu) to utilize NVIDIA CUDA for massive training speedups.

Full CV Pipeline: Managing the complete end-to-end workflow, including data loading (DataLoader), complex image transformations (transforms), model definition, training, and evaluation.

🛠️ Technologies Used

Python 3.x

PyTorch & Torchvision: For building, training, and loading the deep learning model.

NVIDIA CUDA: For GPU acceleration (running on an NVIDIA GeForce RTX 2050).

Scikit-learn (implied): For metrics calculation (though PyTorch handles this).

📈 Methodology

Data Loading: The CIFAR-10 dataset (60,000 32x32 images across 10 classes) is loaded using torchvision.datasets.

Preprocessing: Images are transformed to meet the requirements of the ResNet model: resized to 224x224 and normalized using ImageNet's standard mean and deviation.

Model Definition:

A pre-trained ResNet18 model is loaded from torchvision.models.

All parameters of the model are "frozen" (requires_grad = False) to retain the learned ImageNet features.

The final classification layer (model.fc) is replaced with a new, untrained nn.Linear layer customized for CIFAR-10's 10 classes.

Training (Fine-Tuning):

The model and data are moved to the cuda (GPU) device.

The model is trained for 3 epochs, but only the weights of the new final layer are optimized.

Evaluation: The model's final accuracy is calculated on the unseen test set.

📊 Results

(Catatan: Harap ganti angka di bawah ini dengan hasil aktual dari terminal Anda!)

This demonstrates the immense power of transfer learning and GPU acceleration.

Hardware: NVIDIA GeForce RTX 2050

Average Epoch Time: ~XX.X seconds (Contoh: ~45.2 detik)

Final Test Accuracy: XX.X% (Contoh: 88.7%)

Achieving this level of accuracy in just 3 epochs highlights the effectiveness of fine-tuning a pre-trained model, a task made feasible by the GPU.

🚀 How to Run

Ensure you have a Python environment with PyTorch and Torchvision installed (e.g., the torch-gpu conda environment).

Navigate to the project directory.

Run the script:

python 08_CV_Transfer_Learning.py
