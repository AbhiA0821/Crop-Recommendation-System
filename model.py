import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

X = data.drop('label', axis=1)
y = data['label']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler & encoder
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(7,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(22, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

# Save model
model.save("crop_model.h5")

print("Model + scaler + encoder saved ✅")