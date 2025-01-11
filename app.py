from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model using joblib
model_filename = "classification_model.pkl"
model = joblib.load(model_filename)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Parse input features (e.g., from JSON body)
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
