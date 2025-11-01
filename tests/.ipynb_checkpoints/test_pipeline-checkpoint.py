import pandas as pd
import pytest
from joblib import load
import os
import mlflow
import mlflow.sklearn

# --- MLflow Configuration ---
# This environment variable will be set by the GitHub Action
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") 
REGISTERED_MODEL_NAME = "Iris_Decision_Tree"
MODEL_STAGE = "Staging"

# Define data path (still managed by DVC)
DATA_PATH = "data/eval.csv"

@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Fixture to load the evaluation dataset."""
    if not os.path.exists(DATA_PATH):
        pytest.fail(f"Data file not found at {DATA_PATH}. Run 'dvc pull' first.")
    return pd.read_csv(DATA_PATH)

@pytest.fixture(scope="module")
def model():
    """Fixture to load the model from the MLflow Model Registry."""
    if not MLFLOW_TRACKING_URI:
        pytest.fail("MLFLOW_TRACKING_URI environment variable not set. Cannot find MLflow server.")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    
    try:
        # Load the model from the MLflow Model Registry
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        pytest.fail(f"Failed to load model from MLflow Registry: {e}")

# --- Your existing tests (now working with the MLflow model) ---

def test_data_columns(data):
    """Test 1: Check if the dataset has the expected columns."""
    expected_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    assert expected_columns.issubset(set(data.columns))

def test_data_no_nulls(data):
    """Test 2: Check for no missing values."""
    critical_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    assert data[critical_features].isnull().sum().sum() == 0

def test_model_can_predict(model, data):
    """Test 3: Check if the model can make a prediction."""
    sample = data.sample(1)
    features = sample.drop('species', axis=1)
    features = features[model.feature_names_in_]
    prediction = model.predict(features)
    assert prediction is not None

def test_model_performance(model, data):
    """Test 4: Check model accuracy."""
    X_eval = data.drop('species', axis=1)
    y_true = data['species']
    X_eval = X_eval[model.feature_names_in_]
    y_pred = model.predict(X_eval)
    accuracy = (y_pred == y_true).mean()
    print(f"Model accuracy on eval set: {accuracy:.2f}")
    assert accuracy > 0.9, f"Model accuracy {accuracy:.2f} is below the 0.9 threshold."