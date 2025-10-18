import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

# This script expects the model and evaluation data to be present locally.
# DVC will be responsible for pulling these files.
MODEL_PATH = "local_artifacts/model.joblib"
EVAL_DATA_PATH = "data/eval.csv"

# --- 1. Load Model and Data ---
print("--- Starting Inference ---")
try:
    model = joblib.load(MODEL_PATH)
    eval_df = pd.read_csv(EVAL_DATA_PATH)
    print(" Model and evaluation data loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Make sure to run 'dvc pull' first.")
    sys.exit(1)

X_eval = eval_df.drop("species", axis=1)
y_eval = eval_df["species"]

# --- 2. Run Inference and Show Results ---
print("Running predictions...")
predictions = model.predict(X_eval)

accuracy = accuracy_score(y_eval, predictions)
report = classification_report(y_eval, predictions)

print("\n--- Inference Results ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("--- Inference Complete ---")