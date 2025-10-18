import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
from datetime import datetime

# --- 1. Configuration ---
PROJECT_ID = "sanguine-stock-473303-b4"
DATA_BUCKET_URI = "gs://21f3000274-week1-data"
ARTIFACTS_BUCKET_URI = "gs://21f3000274-week2-artifacts"

# --- 2. Download and Split the Dataset ---
def prepare_data():
    """Downloads the master dataset from GCS and splits it into train/eval sets."""
    print("---  Starting Data Preparation ---")
    
    # Create local data directory
    os.makedirs("data", exist_ok=True)
    master_data_path = "data/data.csv"

    # Download the file from GCS using subprocess
    print(f"Downloading data from {DATA_BUCKET_URI}...")
    subprocess.run(["gsutil", "cp", f"{DATA_BUCKET_URI}/data.csv", master_data_path], check=True)
    
    # Split the data
    print("Splitting data into train and eval sets...")
    df = pd.read_csv(master_data_path)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['species'])

    # Save the splits to be used by DVC and training
    train_path = "data/train.csv"
    eval_path = "data/eval.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    
    print(f" Data preparation complete. Files created: {train_path}, {eval_path}")
    return train_path

# --- 3. Train the Model ---
def train_model(training_data_path):
    """Trains a Decision Tree model on the provided training data."""
    print("\n---  Starting Model Training ---")
    
    # Load ONLY the training data
    training_data = pd.read_csv(training_data_path)
    X_train = training_data.drop('species', axis=1)
    y_train = training_data['species']

    # Train the model
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)
    
    print(" Model training complete.")
    return model

# --- 4. Save and Upload the Model ---
def save_and_upload_model(model):
    """Saves the model locally and uploads it to a timestamped GCS folder."""
    print("\n---  Saving and Uploading Model ---")
    
    os.makedirs("local_artifacts", exist_ok=True)
    local_model_path = "local_artifacts/model.joblib"
    joblib.dump(model, local_model_path)
    print(f"Model saved locally to: {local_model_path}")
    
    # Create a timestamped GCS path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    gcs_model_upload_path = f"{ARTIFACTS_BUCKET_URI}/iris_classifier/{timestamp}/model.joblib"
    
    # Upload to GCS
    print(f"Uploading model to {gcs_model_upload_path}...")
    subprocess.run(["gsutil", "cp", local_model_path, gcs_model_upload_path], check=True)
    
    print(f" Model successfully uploaded.")
    print(f" Use this timestamp for remote inference: {timestamp}")

# --- Main execution block ---
if __name__ == "__main__":
    # This block will run when you execute `python train.py`
    
    # Step 1: Prepare the data
    train_data_file = prepare_data()
    
    # Step 2: Train the model
    trained_model = train_model(train_data_file)
    
    # Step 3: Save and upload the final model
    save_and_upload_model(trained_model)
    
    print("\n Training pipeline finished successfully!")