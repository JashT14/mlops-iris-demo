from flask import Flask, request, jsonify
import joblib
import logging
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "iris_model.pkl"))

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "ml_api.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "MLOps API running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = data["features"]

        if len(features) != 4:
            return jsonify({"error": "Provide exactly 4 features"}), 400

        prediction = model.predict([features])[0]

        logging.info(
            "input=%s prediction=%s",
            features, prediction
        )

        return jsonify({"prediction": str(prediction)})

    except Exception as exc:
        logging.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
