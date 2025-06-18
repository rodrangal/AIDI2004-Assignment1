from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("titanic_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([
        data["Pclass"],
        data["Sex"],
        data["Age"],
        data["SibSp"],
        data["Fare"]
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return jsonify({"survived": bool(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)