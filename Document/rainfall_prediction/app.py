from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "rainfall.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scale.pkl"), "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features = np.array([input_features])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        return render_template("chance.html")
    else:
        return render_template("nochance.html")

if __name__ == "__main__":
    app.run(debug=True)