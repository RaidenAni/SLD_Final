#Import libraries
import os
import cv2 as cv

DATA_DIR = 'Dataset_Prototype_1'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

#Sign Image Display
Sign_img = "Prototype_1\signs.jpg"
Sign_img_display = cv.imread(Sign_img)
cv.imshow("Sign Language", Sign_img_display)


cap = cv.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        if cv.waitKey(20) == ord('s'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        cv.waitKey(25)
        cv.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv.destroyAllWindows()