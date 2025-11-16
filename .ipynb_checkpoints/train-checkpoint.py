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
import numpy as np

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
    return train_path, eval_path

# --- CORRECTED POISONING FUNCTION ---
def poison_data(data_df, poison_level):
    """Poisons a percentage of the training data by replacing features with random noise."""
    if poison_level == 0.0:
        return data_df  # Return the clean data for the 0% baseline run

    print(f"  ...poisoning {poison_level*100}% of training data...")
    poisoned_df = data_df.copy()
    
    n_samples = int(poisoned_df.shape[0] * poison_level)
    if n_samples == 0:
        return poisoned_df
        
    poison_indices = np.random.choice(poisoned_df.index, n_samples, replace=False)
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for col in feature_cols:
        # Create random noise (e.g., from 0 to 50)
        noise = np.random.uniform(0, 50, n_samples)
        
        # Assign the noise as a raw list to avoid index-alignment issues
        poisoned_df.loc[poison_indices, col] = noise.tolist()
        
    return poisoned_df

# --- 3. Training Loop with Hyperparameter Tuning & MLflow ---
def training_pipeline(train_path, eval_path):
    """
    Trains multiple models with different hyperparameters,
    evaluates them, and logs results to MLflow.
    """
    print("\n--- Starting Training Pipeline ---")
    train_data = pd.read_csv(train_path)
    eval_data = pd.read_csv(eval_path)

    X_eval = eval_data.drop('species', axis=1)
    y_eval = eval_data['species']

    poison_levels = [0.0, 0.05, 0.10, 0.50]
    max_depth_options = [7]
    min_samples_split_options = [10]

    # Set the tracking URI FIRST to connect to the remote server.
    mlflow.set_tracking_uri("http://34.122.196.63:8100") # ⚠️ Make sure this IP is correct!
    
    # --- MODIFIED: Use a new experiment name ---
    mlflow.set_experiment("Iris Poisoning v2") 

    for poison_level in poison_levels:
        
        print(f"\n--- Starting run for POISON_LEVEL: {poison_level*100}% ---")
        
        poisoned_train_data = poison_data(train_data.copy(), poison_level)
        X_train = poisoned_train_data.drop('species', axis=1)
        y_train = poisoned_train_data['species']

        for depth in max_depth_options:
            for min_split in min_samples_split_options:
                with mlflow.start_run():
                    print(f"\nTraining with poison={poison_level*100}%, max_depth={depth}, min_split={min_split}...")

                    mlflow.log_param("poison_percentage", poison_level * 100)
                    mlflow.log_param("max_depth", depth)
                    mlflow.log_param("min_samples_split", min_split)

                    model = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=min_split,
                        random_state=1
                    )
                    model.fit(X_train, y_train) 

                    predictions = model.predict(X_eval)
                    accuracy = accuracy_score(y_eval, predictions)
                    print(f"  Accuracy: {accuracy:.4f}")

                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.sklearn.log_model(model, "model")

                    print(f"  ✅ Run logged: poison={poison_level*100}%, depth={depth}, min_split={min_split}, accuracy={accuracy:.4f}")

# --- Main execution block ---
if __name__ == "__main__":
    if not (os.path.exists("data/train.csv") and os.path.exists("data/eval.csv")):
        train_data_file, eval_data_file = prepare_data()
    else:
        train_data_file, eval_data_file = "data/train.csv", "data/eval.csv"
        print("Train/Eval data files already exist.")

    training_pipeline(train_data_file, eval_data_file)
    print("\nTraining pipeline finished successfully! Check MLflow UI.")