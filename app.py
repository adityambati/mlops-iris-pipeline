import os
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load

# Initialize the Flask app
app = Flask(__name__)

# Define the path to the model file
# This file will be placed here by our CD pipeline
MODEL_PATH = "model.joblib"
model = None

@app.before_request
def load_model():
    """Load the model at the start of the first request."""
    global model
    if model is None:
        try:
            model = load(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except FileNotFoundError:
            print(f" ERROR: Model file not found at {MODEL_PATH}.")
        except Exception as e:
            print(f" ERROR: An error occurred loading the model: {e}")

@app.route("/")
def home():
    return "Iris Classifier API is running. Use the /predict endpoint for predictions."

@app.route("/predict", methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    try:
        data = request.get_json(force=True)
        # Assumes input like: {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        df = pd.DataFrame([data])
        
        # Ensure column order matches model training
        df = df[model.feature_names_in_]
        
        prediction = model.predict(df)
        
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))