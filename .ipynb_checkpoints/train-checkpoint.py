import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime
import mlflow
import mlflow.sklearn

# --- 1. Configuration ---
PROJECT_ID = "sanguine-stock-473303-b4"
DATA_BUCKET_URI = "gs://21f3000274-week1-data"
# ARTIFACTS_BUCKET_URI = "gs://21f3000274-week2-artifacts" # No longer needed

# --- 2. Download and Split the Dataset ---
def prepare_data():
    """Downloads the master dataset from GCS and splits it into train/eval sets."""
    print("--- Starting Data Preparation ---")

    os.makedirs("data", exist_ok=True)
    master_data_path = "data/data.csv"
    train_path = "data/train.csv"
    eval_path = "data/eval.csv"

    # Download only if data doesn't exist
    if not os.path.exists(master_data_path):
        print(f"Downloading data from {DATA_BUCKET_URI}...")
        subprocess.run(["gsutil", "cp", f"{DATA_BUCKET_URI}/data.csv", master_data_path], check=True)
    else:
        print("Master data file already exists.")

    print("Splitting data into train and eval sets...")
    df = pd.read_csv(master_data_path)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['species'])
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"✅ Data preparation complete. Files created: {train_path}, {eval_path}")
    # Return both paths
    return train_path, eval_path

# --- 3. Training Loop with Hyperparameter Tuning & MLflow ---
def training_pipeline(train_path, eval_path):
    """
    Trains multiple models with different hyperparameters,
    evaluates them, and logs results to MLflow.
    """
    print("\n--- Starting Training Pipeline ---")
    train_data = pd.read_csv(train_path)
    eval_data = pd.read_csv(eval_path)

    X_train = train_data.drop('species', axis=1)
    y_train = train_data['species']
    X_eval = eval_data.drop('species', axis=1)
    y_eval = eval_data['species']

    max_depth_options = [7]
    min_samples_split_options = [10]

    # --- CORRECTED ORDER ---
    # 1. Set the tracking URI FIRST to connect to the remote server.
    mlflow.set_tracking_uri("http://136.115.81.61:8100") 
    
    # 2. Set the experiment. This will now create the experiment on the remote server.
    mlflow.set_experiment("Iris Classifier Tuning")

    # --- MODIFIED: Nested loop for hyperparameter combinations ---
    for depth in max_depth_options:
        for min_split in min_samples_split_options:
            # Start a new MLflow run for each combination
            with mlflow.start_run():
                print(f"\nTraining with max_depth={depth}, min_samples_split={min_split}...")

                # --- MODIFIED: Log BOTH hyperparameters ---
                mlflow.log_param("max_depth", depth)
                mlflow.log_param("min_samples_split", min_split)

                # --- MODIFIED: Train the model with BOTH hyperparameters ---
                model = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_split=min_split, # Pass the new parameter
                    random_state=1
                )
                model.fit(X_train, y_train)

                # Evaluate the model
                predictions = model.predict(X_eval)
                accuracy = accuracy_score(y_eval, predictions)
                print(f"  Accuracy: {accuracy:.4f}")

                # Log the evaluation metric
                mlflow.log_metric("accuracy", accuracy)

                # Log the trained model artifact
                mlflow.sklearn.log_model(model, "model")

                print(f"  ✅ Run logged: depth={depth}, min_split={min_split}, accuracy={accuracy:.4f}")

# --- Main execution block ---
if __name__ == "__main__":
    if not (os.path.exists("data/train.csv") and os.path.exists("data/eval.csv")):
        train_data_file, eval_data_file = prepare_data()
    else:
        train_data_file, eval_data_file = "data/train.csv", "data/eval.csv"
        print("Train/Eval data files already exist.")

    training_pipeline(train_data_file, eval_data_file)
    print("\nTraining pipeline finished successfully! Check MLflow UI.")