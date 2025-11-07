# Portfolio Project 3: End-to-End Data Modeling - Telco Churn Prediction
# Objective: Demonstrate a full Scikit-learn workflow (EDA, Preprocessing, Modeling, Evaluation).
# This model will later be deployed using Streamlit/Flask.

import pandas as pd
import numpy as np # <-- FIX: Added missing NumPy import
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib # Library for saving the model

# --- 1. MOCK DATA LOADING (In real life, this would be pd.read_csv('Telco.csv')) ---
# We create a synthetic, clean dataset representative of a churn problem
print("1. Loading Data...")
data = {
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'] * 200,
    'SeniorCitizen': [0, 1, 0, 0, 1] * 200,
    'MonthlyCharges': [70.35, 99.65, 104.80, 56.95, 75.20] * 200,
    'TotalCharges': [405.6, 686.4, 765.2, 590.1, 780.4] * 200,
    'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'] * 200,
    'Churn': ['No', 'Yes', 'No', 'Yes', 'No'] * 200
}
df = pd.DataFrame(data)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['MonthlyCharges'] = df['MonthlyCharges'] + np.random.randn(len(df)) * 5 # Add noise

print(f"Dataset loaded: {len(df)} rows.")

# --- 2. DATA PREPROCESSING AND FEATURE ENGINEERING ---

# A. Handle Categorical Features (One-Hot Encoding for 'Gender' and 'Contract')
print("2. Preprocessing Data (One-Hot Encoding)...")
df = pd.get_dummies(df, columns=['Gender', 'Contract'], drop_first=True)

# B. Target Variable Encoding (Already done above, but good to check)
# df['Churn'] is already 0 (No) or 1 (Yes)

# C. Define Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- 3. TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- 4. MODEL TRAINING (Random Forest Classifier) ---
# Random Forest is powerful and requires minimal feature scaling.
print("3. Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training complete.")

# --- 5. MODEL EVALUATION ---
print("4. Evaluating Model Performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# --- 6. MODEL EXPORT (Saving the Model for Future Deployment) ---
model_filename = 'churn_predictor_rf.joblib'
joblib.dump(model, model_filename)
print(f"\nModel successfully saved to: {model_filename}")

# --- 7. GPU CHECK (Passive check for future use, should be silent here) ---
try:
    import torch
    if torch.cuda.is_available():
        print("\nNote: PyTorch is installed and CUDA is available for Deep Learning projects.")
except:
    pass