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

# --- CORRECTED POISONING FUNCTION (Label Flipping) ---
def poison_data(data_df, poison_level):
    """
    Poisons a percentage of the training data by FLIPPING THE LABELS.
    """
    if poison_level == 0.0:
        return data_df  # Return the clean data

    print(f"  ...poisoning {poison_level*100}% of training LABELS...")
    poisoned_df = data_df.copy()
    
    n_samples = int(poisoned_df.shape[0] * poison_level)
    if n_samples == 0:
        return poisoned_df
        
    poison_indices = np.random.choice(poisoned_df.index, n_samples, replace=False)
    
    all_species = ['setosa', 'versicolor', 'virginica']
    
    for i in poison_indices:
        original_label = poisoned_df.loc[i, 'species']
        wrong_labels = [label for label in all_species if label != original_label]
        poisoned_df.loc[i, 'species'] = np.random.choice(wrong_labels)
            
    return poisoned_df

# --- 3. Training Loop with Data Poisoning ---
def training_pipeline(train_path, eval_path):
    """
    Trains one model for each poison level and logs to MLflow.
    """
    print("\n--- Starting Training Pipeline ---")
    train_data = pd.read_csv(train_path)
    eval_data = pd.read_csv(eval_path)

    X_eval = eval_data.drop('species', axis=1)
    y_eval = eval_data['species']

    # Define poison levels from the assignment
    poison_levels = [0.0, 0.05, 0.10, 0.50]
    
    # --- SIMPLIFIED: Use one set of good hyperparameters ---
    fixed_depth = 7
    fixed_min_split = 10

    # 1. Set the tracking URI FIRST to connect to the remote server.
    mlflow.set_tracking_uri("http://34.122.196.63:8100") # ⚠️ Make sure this IP is correct!
    
    # 2. Set the experiment.
    mlflow.set_experiment("Iris Poisoning - Label Flip Attack")

    # --- SIMPLIFIED: Loop ONLY through poison levels ---
    for poison_level in poison_levels:
        
        print(f"\n--- Starting run for POISON_LEVEL: {poison_level*100}% ---")
        
        poisoned_train_data = poison_data(train_data.copy(), poison_level)
        X_train = poisoned_train_data.drop('species', axis=1)
        y_train = poisoned_train_data['species']

        # Start a new MLflow run
        with mlflow.start_run():
            print(f"\nTraining with poison={poison_level*100}%...")

            # Log parameters
            mlflow.log_param("poison_percentage", poison_level * 100)
            mlflow.log_param("max_depth", fixed_depth)
            mlflow.log_param("min_samples_split", fixed_min_split)

            model = DecisionTreeClassifier(
                max_depth=fixed_depth,
                min_samples_split=fixed_min_split,
                random_state=1
            )
            model.fit(X_train, y_train) 

            # Evaluate the model on the CLEAN evaluation set
            predictions = model.predict(X_eval)
            accuracy = accuracy_score(y_eval, predictions)
            print(f"  Accuracy: {accuracy:.4f}")

            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")

            print(f"  ✅ Run logged: poison={poison_level*100}%, accuracy={accuracy:.4f}")

# --- Main execution block ---
if __name__ == "__main__":
    if not (os.path.exists("data/train.csv") and os.path.exists("data/eval.csv")):
        train_data_file, eval_data_file = prepare_data()
    else:
        train_data_file, eval_data_file = "data/train.csv", "data/eval.csv"
        print("Train/Eval data files already exist.")

    training_pipeline(train_data_file, eval_data_file)
    print("\nTraining pipeline finished successfully! Check MLflow UI.")