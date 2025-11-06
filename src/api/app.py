from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/fraud_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict_proba(df)[:, 1][0]
    return jsonify({"fraud_probability": round(float(prediction), 3)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
