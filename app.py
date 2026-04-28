from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load files
model = load_model("crop_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]
    final = scaler.transform([values])

    prediction = model.predict(final)
    crop = le.inverse_transform([np.argmax(prediction)])

    return render_template("index.html", prediction_text=f"Recommended Crop: {crop[0]}")

if __name__ == "__main__":
    app.run(debug=True)