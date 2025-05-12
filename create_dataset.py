import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = r"C:\Users\karthi\Desktop\marcus_project\data"

# Check if the directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: DATA_DIR '{DATA_DIR}' does not exist.")
    exit()

print("Data collection started...")

# Debugging: Print directory structure
print("Classes found:", os.listdir(DATA_DIR))

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        print(f"Skipping {dir_}, not a directory.")
        continue
    
    print(f"Processing class: {dir_}")
    
    for img_path in os.listdir(class_path):
        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(class_path, img_path)
        img = cv2.imread(img_full_path)
        
        if img is None:
            print(f"Warning: Unable to read image {img_full_path}")
            continue  # Skip unreadable images

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            
            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"No hands detected in {img_path}")

# Save collected data
if data:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("Data collection completed successfully. Data saved in 'data.pickle'")
    print(f"Total samples collected: {len(data)}")
else:
    print("Error: No valid data collected.")
