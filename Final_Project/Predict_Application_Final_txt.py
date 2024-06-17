# Import Libraries
import cv2
import numpy as np
import mediapipe as mp
import pickle
import tkinter as tk
import tkinter.filedialog as fd 

from PIL import Image, ImageTk

# Load the model 
model_dict = pickle.load(open('Final_Project\model_final.p', 'rb'))
model = model_dict['model']

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define Labels and Letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)  # Start capturing video
        self.current_image = None
    
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Sign Language")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("2560x1600") # Set window size
        
        # Create labels for displaying video, characters, words, and sentences
        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 180, width = 650, height = 580)
        
        self.panel3 = tk.Label(self.root) # Current Symbol (Character)
        self.panel3.place(x = 400, y = 735)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 100, y = 735)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))
        
        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 250, y = 780)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 100,y = 780)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))
        
        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 400, y = 830)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 100, y = 830)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))
        
        # Load and display sign language image
        image = Image.open("Final_Project\signs_800x444.jpg") #Sign Image
        Sign_image = ImageTk.PhotoImage(image)
        label_image = tk.Label(image = Sign_image)
        label_image.image = Sign_image
        label_image.place(x = 800, y = 240)
        
        # Load and display header image
        image_header = Image.open("Final_Project\Header_1033x200.png") #Sign Image
        Header_image = ImageTk.PhotoImage(image_header)
        label_image = tk.Label(image = Header_image)
        label_image.image = Header_image
        label_image.place(x = 350, y = 0)
        
        # Variables for storing text and predictions
        self.str = ""
        self.word = ""
        self.current_symbol = ""
        self.photo = ""

        # Add a flag to track if file is saved
        self.file_saved = False  
        self.first_save = False
        
        # Start the video loop
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read() # Capture frame from camera

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
                    x_ = []  # Store x coordinates
                    y_ = []  # Store y coordinates
                    data_aux = []

                    # First loop to calculate min(x_) and min(y_)
                    for i in range(21):  # All 21 landmarks
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

                    prediction = model.predict([np.asarray(data_aux)]) # Predict the sign using the trained model
                    
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
                    
    
                    else:
                        self.panel3.config(text="Blank", font=("Courier", 30))
                        
            # Convert the processed frame to ImageTk and update
            if not hand_detected: # Check hand_detected here, after processing all hands
                self.panel3.config(text="Blank", font=("Courier", 30))

            # Update the video display
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.panel.configure(image=image)
            self.panel.image = image

        self.root.after(10, self.video_loop) 

    # Key press events
    def on_key_press(self, event):
            if event.char == 's':  # Save to Word
                self.word += self.current_symbol
                self.panel4.config(text=self.word, font=("Courier", 30))
                
            elif event.char == ' ':  # Sentence formation
                if self.word != "":  # Add word only if it's not empty
                    self.str += self.word + " "
                    self.panel5.config(text=self.str, font=("Courier", 30))
                    self.word = ""  # Reset word after adding to sentence
                    self.panel4.config(text=self.word, font=("Courier", 30))
                    
            elif event.char == 'd':  # Delete last character from Word
                if self.word:  # Check if the word is not empty
                    self.word = self.word[:-1]  # Remove the last character
                    self.panel4.config(text=self.word, font=("Courier", 30))  # Update the labe
                    
            elif event.keysym == 'BackSpace':  # Delete last character from sentence
                if self.str:
                    self.str = self.str[:-1]
                    self.panel5.config(text=self.str, font=("Courier", 30))

    # Button "Save" 
    def save_sentence(self):
        filepath = fd.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filepath:
            self.save_filepath = filepath
            with open(filepath, "w") as file:
                file.write(self.str)
            self.file_saved = True  
            self.update.config(state=tk.NORMAL)
            self.str = ""  # Clear the sentence after saving
            self.panel5.config(text=self.str, font=("Courier", 30))  # Update the Sentence label
        self.first_save = True

    # Button "Update"
    def update_file(self):
        if self.file_saved and self.str:
            try:
                with open(self.save_filepath, "a") as file:
                    # Append newline ONLY if it's NOT the first save
                    if not self.first_save:
                        file.write("\n")
                    file.write(self.str) 

                self.clear_sentence()
                print("Sentence appended successfully!")

                # Reset the first_save flag after updating
                self.first_save = False  
            except Exception as e:
                print("Error updating file:", e)

            
    # Button "Clear"
    def clear_sentence(self):
        self.str = ""
        self.panel5.config(text=self.str, font=("Courier", 30))
    
    def destructor(self):
        
        print("Closing Application...")
        
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Opening...")

# Create the application instance
app = Application()

# Design Save button
app.save = tk.Button(app.root)
app.save.place(x=1005, y=875)
app.save.config(text="Save", font=("Courier", 20), wraplength=100, command=app.save_sentence)

# Design Update button 
app.update = tk.Button(app.root)
app.update.place(x=1200, y=875) 
app.update.config(text="Update", font=("Courier", 20), wraplength=100, command=app.update_file, state=tk.DISABLED) # Initially disable the Update button

# Design Clear button
app.clear = tk.Button(app.root)
app.clear.place(x=1400, y=875)
app.clear.config(text="Clear", font=("Courier", 20), wraplength=100, command=app.clear_sentence)

# Bind key press events to the function
app.root.bind("<Key>", app.on_key_press)

# Start the main event loop
app.root.mainloop()