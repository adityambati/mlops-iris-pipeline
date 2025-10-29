import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
import mlflow
import mlflow.sklearn

# --- Configuration ---
# Define the MLflow model URI using the model registry
# Format: "models:/<registered_model_name>/<stage_or_version>"
REGISTERED_MODEL_NAME = "Iris_Decision_Tree" # Choose a name for your registered model
MODEL_STAGE = "Staging" # Or "Production" or a specific version number
model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"

# Data path (still managed by DVC)
EVAL_DATA_PATH = "data/eval.csv"

# --- 1. Load Model from MLflow Registry ---
print("--- Starting Inference ---")
print(f"Loading model '{REGISTERED_MODEL_NAME}' stage '{MODEL_STAGE}' from MLflow Registry...")
try:
    # Set tracking URI if your MLflow server is running elsewhere (not needed for local)
    # mlflow.set_tracking_uri("http://YOUR_SERVER_IP:8100") 
    model = mlflow.sklearn.load_model(model_uri)
    print("✅ Model loaded successfully from MLflow Registry.")
except Exception as e:
    print(f"❌ Error loading model from MLflow Registry: {e}")
    print("Ensure the model is registered and promoted to the correct stage in MLflow UI.")
    sys.exit(1)

# --- 2. Load Evaluation Data (from DVC) ---
try:
    eval_df = pd.read_csv(EVAL_DATA_PATH)
    print(f"✅ Evaluation data loaded from {EVAL_DATA_PATH}.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}. Make sure to run 'dvc pull data/eval.csv' first.")
    sys.exit(1)

X_eval = eval_df.drop("species", axis=1)
y_eval = eval_df["species"]

# --- 3. Run Inference and Show Results ---
print("Running predictions...")
# Ensure columns are in the correct order the model expects
X_eval = X_eval[model.feature_names_in_]
predictions = model.predict(X_eval)

accuracy = accuracy_score(y_eval, predictions)
report = classification_report(y_eval, predictions)

print("\n--- Inference Results ---")
print(f"Using Model: {model_uri}")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("--- Inference Complete ---")