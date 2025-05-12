import os
import cv2

DATA_DIR = r"C:\Users\Lenovo\Desktop\marcus\project\data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 29  # Number of different classes
dataset_size = 150  # Change from 100 to 150

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user to be ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        
        # Save image
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
