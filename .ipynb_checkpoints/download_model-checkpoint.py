import os
import mlflow
import joblib

print("--- Starting model download from MLflow Registry ---")

# 1. Get the MLflow server URI from the environment variable
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("FATAL: MLFLOW_TRACKING_URI environment variable not set.")
    exit(1)

mlflow.set_tracking_uri(tracking_uri)
print(f"Set MLflow tracking URI to: {tracking_uri}")

# 2. Define the model to download
model_uri = "models:/Iris_Decision_Tree/Staging"
print(f"Loading model from: {model_uri}")

try:
    # 3. Load the model from the registry
    model = mlflow.sklearn.load_model(model_uri)
    
    # 4. Save the model to a local file
    joblib.dump(model, 'model.joblib')
    
    print("✅ Model downloaded from MLflow Registry and saved as model.joblib")

except Exception as e:
    print(f"FATAL: An error occurred while downloading the model: {e}")
    exit(1)