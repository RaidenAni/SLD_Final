# Import Libraries 
import cv2
import numpy as np
import mediapipe as mp
import os
import string

# Define the directory
directory = 'Dataset_Prototype_2/'

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5) # Static mode for single image processing (Này trên trang chủ Mediapipe nó có nói từng thông số là gì)

# Create folders for each character (A (0) -> Z (25)) 
for i in range(26):
    folder_name = os.path.join(directory, str(i))  
    os.makedirs(folder_name, exist_ok=True)      

#Sign Image Display
Sign_img = "Final_Project/signs_800x444.jpg"
Sign_img_display = cv2.imread(Sign_img)
cv2.imshow("Sign Language", Sign_img_display)

# Start video capture
capture = cv2.VideoCapture(0)
interrupt = -1

while True:
    # Capture a frame 
    _, frame = capture.read()

    # Flip the frame
    frame = cv2.flip(frame, 1)
    # Copy of the original frame
    original_frame = frame.copy()
    
    # Count existing images
    count = {}
    for i in range(26):  
        folder_name = os.path.join(directory, str(i))  
        if os.path.exists(folder_name):
            count[chr(i + 97)] = len(os.listdir(folder_name))  
        else:
            count[chr(i + 97)] = 0  

    # Display character counts on the screen
    for i, char in enumerate(string.ascii_lowercase):
        cv2.putText(frame, f"{char.upper()} ({i}): {count[char]}", (10, 70 + 10 * i), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    # Convert frame to RGB for MediaPipe processing
    results = hands.process(frame)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 

    # If hand landmarks are detected, draw them on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec,
                connection_drawing_spec
            )

    # Display the frame with landmarks
    cv2.imshow("Frame", frame)
    
    #Key press 
    key = cv2.waitKey(10) # Delay 10ms
    if key & 0xFF == 27:  # Exit on Esc key
        break
    elif key & 0xFF in map(ord, string.ascii_lowercase):
        char = chr(key & 0xFF)
        index = ord(char) - 97

        # Save the ORIGINAL frame 
        cv2.imwrite(directory + str(index) + '/' + str(count[char]) + '.jpg', original_frame)

# Release the video capture and close windows
capture.release()
cv2.destroyAllWindows()