# Portfolio Project 2: BERT Sentiment Inference (Testing Script)
# Objective: Load the saved Transformer model and use it to predict the sentiment of new, unseen text.

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os # Import os to check path

# --- Configuration ---
# OTOMATIS mendeteksi GPU Anda. Ini adalah perubahan kuncinya.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
MODEL_PATH = 'NLP_Sentiment_Analysis/bert_sentiment_model' # Path to the saved model folder
LABEL_MAP = {0: "Negative 🔴", 1: "Positive 🟢"} # Memperbaiki emoji

# --- 1. Load Model and Tokenizer ---
print(f"1. Loading saved BERT model and tokenizer...")
print(f"Menggunakan device: {device}")

# Check if the path exists before trying to load
if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: Model folder not found at {MODEL_PATH}")
    print("Please ensure you successfully ran '07_Transformer_Sentiment.py' first to train and save the model.")
    exit()

try:
    # Load the tokenizer (to convert text to numbers)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    
    # Load the saved model (from the folder created after training)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device) # Pindahkan model ke GPU
    model.eval() # Set the model to evaluation mode
    print("Model loaded successfully from disk.")

except Exception as e:
    print(f"\nERROR: Failed to load model from {MODEL_PATH}")
    print(f"Details: {e}")
    exit()

# --- 2. Inference Function ---
def predict_sentiment(text):
    """Tokenizes text, runs inference, and returns prediction."""
    
    encoded_input = tokenizer(text, 
                              padding='max_length', 
                              truncation=True, 
                              max_length=128, 
                              return_tensors='pt')
    
    # Pindahkan data input ke GPU
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Pindahkan hasil kembali ke CPU untuk di-decode
    probabilities = F.softmax(logits, dim=1).cpu()
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    
    return LABEL_MAP[predicted_class], confidence

# --- 3. Test Cases ---
print("\n--- 2. Running Real-Time Sentiment Predictions ---")

test_sentences = [
    "This movie was an absolute masterpiece; the acting was superb and the ending was moving.",
    "The film was boring and the plot made no sense. I walked out halfway through.",
    "It was okay, not the best, but I wouldn't call it terrible either."
]

for i, sentence in enumerate(test_sentences):
    sentiment, confidence = predict_sentiment(sentence)
    print(f"\n[{i+1}] TEXT: '{sentence[:60]}...'")
    print(f"    PREDICTION: {sentiment} (Confidence: {confidence:.4f})")

print("\nInference Complete.")