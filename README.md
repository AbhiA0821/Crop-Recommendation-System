# 🌱 Crop Recommendation System using Deep Learning

This project is a web-based application that recommends the most suitable crop based on soil nutrients and environmental conditions. It uses a Deep Learning model (Artificial Neural Network) to make accurate predictions and is deployed using Flask.

---

## 📌 Problem Statement
Farmers often face difficulty in selecting the right crop due to lack of knowledge about soil composition and environmental conditions. This can lead to low yield and financial loss.

---

## 💡 Solution
This system analyzes key agricultural parameters and recommends the best crop using a trained deep learning model.

---

## 🧠 Model Details
- Model Type: Artificial Neural Network (ANN)
- Framework: TensorFlow / Keras
- Activation Functions:
  - ReLU (hidden layers)
  - Softmax (output layer)
- Accuracy: ~95%

---

## 📊 Input Features
The model takes 7 inputs:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH value
- Rainfall

---

## 🌾 Output
- Recommended Crop (from 22 crop types)

---

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- Flask
- Scikit-learn
- Pandas, NumPy
- HTML, CSS (Frontend)

---

## 🌐 Web Application
The model is deployed using Flask, where users can:
1. Enter soil and environmental values
2. Click "Predict"
3. Get instant crop recommendation

---

