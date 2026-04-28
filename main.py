# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("Crop_recommendation.csv")

print("First 5 rows:")
print(data.head())

# =========================
# 2. Prepare Data
# =========================
X = data.drop('label', axis=1)   # Inputs
y = data['label']                # Output

# Convert labels (crop names → numbers)
le = LabelEncoder()
y = le.fit_transform(y)

# Scale input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Build Model
# =========================
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(7,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(22, activation='softmax')  # 22 crop classes
])

# =========================
# 5. Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 6. Train Model
# =========================
history = model.fit(X_train, y_train, epochs=50, batch_size=16)

# =========================
# 7. Evaluate Model
# =========================
loss, accuracy = model.evaluate(X_test, y_test)
print("\nModel Accuracy:", accuracy)

# =========================
# 8. Make Prediction
# =========================
sample = np.array([[90, 42, 43, 20.5, 80, 6.5, 200]])
sample = scaler.transform(sample)

prediction = model.predict(sample)
predicted_class = np.argmax(prediction)

crop = le.inverse_transform([predicted_class])
print("\nRecommended Crop:", crop[0])

# =========================
# 9. Plot Accuracy Graph
# =========================
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# =========================
# 10. Save Model
# =========================
model.save("crop_model.h5")
print("\nModel saved as crop_model.h5")