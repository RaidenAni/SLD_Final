import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}

import mediapipe as mp
import cv2
import pickle
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)  # Start capturing video
        self.current_image = None

        # Load the trained SVM model
        with open('./Prototype_2/model_3.p', 'rb') as f:
            model_dict = pickle.load(f)
        self.model = model_dict['model']
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Sign Language")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("2560x1600")
        
        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 160, width = 580, height = 580)

        self.panel3 = tk.Label(self.root) # Current Symbol
        self.panel3.place(x = 800, y = 705)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 475, y = 705)
        self.T1.config(text = "Character :", font = ("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 800, y = 750)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 475,y = 750)
        self.T2.config(text = "Word :", font = ("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 800, y = 800)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 475, y = 800)
        self.T3.config(text = "Sentence :",font = ("Courier", 30, "bold"))
        
        image = Image.open("Prototype_2\signs_800x444.jpg") #Sign Image
        Sign_image = ImageTk.PhotoImage(image)
        label_image = tk.Label(image = Sign_image)
        label_image.image = Sign_image
        # Position Sign image
        label_image.place(x = 800, y = 225)
        
        image_header = Image.open("Prototype_2\Header_1033x200.png") #Sign Image
        Header_image = ImageTk.PhotoImage(image_header)
        label_image = tk.Label(image = Header_image)
        label_image.image = Header_image
        # Position Header image
        label_image.place(x = 350, y = 3)
        
        # Initialize text for display
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

        # Start the video loop
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)
            
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            H, W, _ = frame.shape  
            results = hands.process(frame)
            
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

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
                        
                if len(data_aux) == 42:
                    prediction = self.model.predict([np.asarray(data_aux)])
                    self.current_symbol = self.labels_dict[int(prediction[0])]

                    if self.current_symbol == 'space':  # Giả sử 'space' là ký tự cách
                        if self.word:  # Nếu từ không rỗng
                            self.str += self.word + " "
                            self.word = ""  # Đặt lại từ
                    else:
                        self.word += self.current_symbol  # Thêm ký tự vào từ
            else:
                self.current_symbol = "Empty"  # Đặt lại khi không phát hiện tay

                self.panel3.config(text=self.current_symbol, font=("Courier", 30))
                self.panel4.config(text=self.word, font=("Courier", 30))
                self.panel5.config(text=self.str, font=("Courier", 30))
        

        self.root.after(5, self.video_loop)
        
    def destructor(self):
        
        print("Closing Application...")
        
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
