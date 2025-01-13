from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model using joblib
model_filename = "classification_model.pkl"
model = joblib.load(model_filename)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    print("Received data:", data)

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Features without column names
        features = np.array(data["features"]).reshape(1, -1)
        print("Features to predict with:", features)

        # If your model expects a DataFrame, you can do something like:
        feature_names = [
            "gender", "age", "hypertension", "heart_disease", "marital_status", 
            "occupation", "residence", "glucose", "bmi", "smoking_status"
        ]
        features_df = pd.DataFrame(features, columns=feature_names)

        prediction = model.predict(features_df)[0]
        print("Prediction:", prediction)

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
