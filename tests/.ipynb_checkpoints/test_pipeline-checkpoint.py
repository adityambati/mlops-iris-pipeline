import pandas as pd
import pytest
from joblib import load
import os

# Define the paths to the data and model files that DVC will pull.
# These are relative to the project's root directory where the tests will run.
DATA_PATH = "data/eval.csv"
MODEL_PATH = "local_artifacts/model.joblib"

@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Fixture to load the evaluation dataset, failing if the file isn't found."""
    if not os.path.exists(DATA_PATH):
        pytest.fail(f"Data file not found at {DATA_PATH}. Ensure 'dvc pull' has run.")
    return pd.read_csv(DATA_PATH)

@pytest.fixture(scope="module")
def model():
    """Fixture to load the model, failing if the file isn't found."""
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model file not found at {MODEL_PATH}. Ensure 'dvc pull' has run.")
    return load(MODEL_PATH)

def test_data_columns(data):
    """Test 1: Check if the dataset has the expected columns."""
    expected_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    assert expected_columns.issubset(set(data.columns))

def test_data_no_nulls(data):
    """Test 2: Check for no missing values in critical feature columns."""
    critical_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    assert data[critical_features].isnull().sum().sum() == 0

def test_data_species_values(data):
    """Test 3: Ensure the 'species' column has the expected set of values."""
    expected_species = {'setosa', 'versicolor', 'virginica'}
    assert set(data['species'].unique()).issubset(expected_species)

def test_model_can_predict(model, data):
    """Test 4: Check if the loaded model can make a prediction on a sample."""
    # Take a single sample from the data to test prediction
    sample = data.sample(1)
    features = sample.drop('species', axis=1)
    
    # Ensure columns are in the correct order that the model expects
    # This uses the feature_names_in_ attribute stored during model training
    features = features[model.feature_names_in_]
    
    prediction = model.predict(features)
    assert prediction is not None
    assert isinstance(prediction[0], str) # Expecting a string like 'setosa'