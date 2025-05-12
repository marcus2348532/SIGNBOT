import cv2
import mediapipe as mp
import os
import numpy as np
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define the dataset path
DATASET_PATH = r"C:\Users\karthi\Desktop\marcus_project\data2"

# Define all labels extracted from the image
LABELS = ['Afraid', 'Agree', 'Assistance', 'Bad', 'Become', 'College', 'Doctor', 'From', 'Hello', 'I Love You', 'No', 
          'Pain', 'Please', 'Pray', 'Secondary', 'Skin', 'Small', 'Specific', 'Stand', 'Thanks', 'Today', 'Warn', 
          'How', 'Which', 'Work', 'Yes', 'You']

# Prepare storage
X_data = []
y_labels = []

for label in LABELS:
    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder_path):  # Skip missing folders
        print(f"⚠️ Warning: Folder '{label}' not found, skipping...")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:  # Skip unreadable images
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe Hands
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for point in hand_landmarks.landmark:
                    landmarks.extend([point.x, point.y])  # Use only x, y coordinates
                
                X_data.append(landmarks)
                y_labels.append(label)  # Store corresponding label

# Convert to NumPy arrays
X_data = np.array(X_data)
y_labels = np.array(y_labels)

# Save dataset using Pickle
with open("landmark_dataset.p", "wb") as f:
    pickle.dump((X_data, y_labels), f)

print("✅ Dataset extracted and saved as landmark_dataset.p")
