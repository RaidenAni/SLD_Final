import os

import mediapipe as mp
import cv2
import pickle
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Load the model 
model_dict = pickle.load(open('./Prototype_2/model_3.p', 'rb'))
model = model_dict['model']

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)  # Start capturing video
        self.current_image = None
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Sign Language")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("2560x1600")
        
        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 180, width = 650, height = 580)

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 800, y = 735)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 475, y = 735)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 800, y = 780)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 475,y = 780)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 800, y = 830)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 475, y = 830)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))
        
        # Position Sign image
        image = Image.open("Prototype_2\signs_800x444.jpg") #Sign Image
        Sign_image = ImageTk.PhotoImage(image)
        label_image = tk.Label(image = Sign_image)
        label_image.image = Sign_image
        label_image.place(x = 800, y = 240)
        
        
        # Position Header image
        image_header = Image.open("Prototype_2\Header_1033x200.png") #Sign Image
        Header_image = ImageTk.PhotoImage(image_header)
        label_image = tk.Label(image = Header_image)
        label_image.image = Header_image
        label_image.place(x = 350, y = 0)
        
        # Initialize text for display
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
   
        # Call the video loop function
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            # Convert frame to RGB and flip
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)

            # Process the frame with MediaPipe Hands
            results = hands.process(frame)
            hand_detected = False  # Flag to track hand detection

            # If hand landmarks are detected, perform sign recognition
            if results.multi_hand_landmarks:
                hand_detected = True

                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = []  # List to store x coordinates
                    y_ = []  # List to store y coordinates
                    data_aux = []

                    # First loop to calculate min(x_) and min(y_)
                    for i in range(21):  # Iterate over all 21 landmarks
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Second loop to normalize and store in data_aux
                    for i in range(21):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))  # Normalize x
                        data_aux.append(y - min(y_))  # Normalize y

                    prediction = model.predict([np.asarray(data_aux)])
                    
                    if hand_detected:
                        # Get the predicted character
                        predicted_character = labels_dict[int(prediction[0])]
                                        # Update the current symbol
                        if self.current_symbol == predicted_character:
                            self.panel3.config(text=predicted_character, font=("Courier", 30))
                        else:
                            self.current_symbol = predicted_character
                            self.panel3.config(text=predicted_character, font=("Courier", 30))
                        # Update GUI elements with the prediction
                        self.panel3.config(text=predicted_character, font=("Courier", 30))
                    # ... Update self.word, self.str (logic for word/sentence formation)
                    else:
                        self.panel3.config(text="Blank", font=("Courier", 30))
            # Convert the processed frame to ImageTk and update
            if not hand_detected: # Check hand_detected here, after processing all hands
                self.panel3.config(text="Blank", font=("Courier", 30))
            # ... (Optionally clear self.word and self.str)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.panel.configure(image=image)
            self.panel.image = image

        self.root.after(10, self.video_loop) 

    def on_key_press(self, event):
            if event.char == 's':  # Word formation
                self.word += self.current_symbol
                self.panel4.config(text=self.word, font=("Courier", 30))
            elif event.char == ' ':  # Sentence formation
                if self.word != "":  # Add word only if it's not empty
                    self.str += self.word + " "
                    self.panel5.config(text=self.str, font=("Courier", 30))
                    self.word = ""  # Reset word after adding to sentence
                    self.panel4.config(text=self.word, font=("Courier", 30))

    def destructor(self):
        
        print("Closing Application...")
        
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

# Create the application instance
app = Application()

# Bind key press events to the function
app.root.bind("<Key>", app.on_key_press)

# Start the main event loop
app.root.mainloop()