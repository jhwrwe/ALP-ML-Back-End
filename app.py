from flask import Flask, request, jsonify
import joblib
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

    if not data or 'features' not in data:
        return jsonify({"error": "No input data provided or 'features' key missing"}), 400

    try:
        # Extract the 'features' from the request
        features = np.array(data["features"]).reshape(1, -1)
        print("Features to predict with:", features)

        # Define the column names (features) expected by the model
        feature_names = [
            "gender", "age", "hypertension", "heart_disease", "ever_married", 
            "Residence_type", "avg_glucose_level", "bmi", "Private", "Self_employed", "Govt_job", 
            "Never_worked", "children", "formerly_smoked", "never_smoked", "has_smokes", "smoke_Unknown"
        ]

        # Convert the features array to a DataFrame with the expected column names
        features_df = pd.DataFrame(features, columns=feature_names)

        # Use model.predict_proba() to get the probabilities
        probabilities = model.predict_proba(features_df)[0]
        print("Probabilities:", probabilities)

        # The probability for class 1 (index 1)
        prob_class_1 = probabilities[1]
        print("Probability of being class 1:", prob_class_1)

        # Return the probability of class 1
        return jsonify({"probability_class_1": prob_class_1})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
