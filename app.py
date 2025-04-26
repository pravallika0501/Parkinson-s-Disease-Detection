from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        input_scaled = scaler.transform([input_features])
        prediction = model.predict(input_scaled)[0]
        result = "Parkinson's Detected" if prediction == 1 else "Healthy"
        return render_template("index.html", prediction_text=result)
    except:
        return render_template("index.html", prediction_text="Error: Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)
