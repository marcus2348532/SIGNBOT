import pickle
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
with open("landmark_dataset.p", "rb") as f:
    X_data, y_labels = pickle.load(f)

# Encode labels to numeric values
label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(y_labels)  # Converts labels to numbers (0-26)

# Normalize the landmark data
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)  # Standardize feature values

# Reshape data for LSTM (samples, timesteps, features)
X_data = np.array(X_data)
X_data = X_data.reshape((X_data.shape[0], 1, X_data.shape[1]))  # 1 timestep

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Build Enhanced LSTM Model
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(1, X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(27, activation='softmax')  # 27 classes (A-Z + space)
])

# Compile the model with a lower learning rate
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0003), metrics=['accuracy'])

# Train the model with more epochs and larger batch size
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=75, batch_size=64)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("🔢 Confusion Matrix:\n", cm)

# Plot confusion matrix as heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Save the trained model
model.save("signlang_lstm_model_improved.h5")
with open("label_encoder.p", "wb") as f:
    pickle.dump(label_encoder, f)
with open("scaler.p", "wb") as f:
    pickle.dump(scaler, f)  # Save scaler for preprocessing in real-time predictions

print("✅ Improved LSTM model trained and saved as signlang_lstm_model_improved.h5")
