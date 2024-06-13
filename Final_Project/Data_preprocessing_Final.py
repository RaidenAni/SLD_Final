# Import Libraries
import os
import pickle

import cv2 as cv
import mediapipe as mp
import matplotlib.pyplot as plt

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to Dataset
DATA_DIR = './Dataset_Prototype_2'

# Store extracted data and labels
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [] # Temporary list to store coordinates for every single image

        x_ = [] # Store x coordinates
        y_ = [] # store y coordinates

        img = cv.imread(os.path.join(DATA_DIR, dir_, img_path)) # Read the image
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Convert image to RGB format

        # Process the image with MediaPipe  
        results = hands.process(img_rgb)
        
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Store x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            
            data.append(data_aux) # Add the extracted and normalized landmark data to the main data list
            labels.append(dir_) # Add the corresponding label

f = open('./Final_Project/data.pickle_Final', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()