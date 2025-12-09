import cv2
import numpy as np
import os
from pathlib import Path

# Path for authorized face image
authorized_face_path = "authorized.jpg"

# Load Haar Cascade for face detection
# The Haar Cascade model is a pre-trained face detection classifier provided by OpenCV. It scans the image for facial features and returns coordinates of the face. It detects where the face is, not who the face is.
# for ex: [(150, 80, 120, 120)]
# face found at x=150, y=80
# width = 120
# height = 120
# Your code then draws a rectangle around this region.

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Desktop folder path
desktop_path = str(Path.home() / "Desktop") # "C:\Users\Yash\Desktop"

# Function to compare faces using simple pixel difference
def compare_faces(face1, face2):
    face1 = cv2.resize(face1, (100, 100)) # face1 = Stored Image i.e authorized.jpg
    face2 = cv2.resize(face2, (100, 100)) # face2 = Web cam captured it, After detecting it returns the coordinate convert in gray
    diff = np.sum(np.abs(face1 - face2)) # If value is -ve it converts in +ve with no change in value like absolute mean error
    return diff

# Start webcam
cap = cv2.VideoCapture(0)

print("Press S to save your face (first time).")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Converts the color frame (BGR) into grayscale.
    # Face detection & comparison both use grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray image using above model face detector. from above model import
    # 2️⃣ What is .detectMultiScale()?
    # This function scans the image and finds all faces in it.
    # Meaning:
    # 👉 It cuts the image into small blocks = (gray, 1.3, 5)
    # 👉 Checks each part
    # 👉 If that part looks like a face → it returns the position
    # So .detectMultiScale() is the function that finds faces.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # It returns a (x, y, w, h), 

    for (x, y, w, h) in faces: # Loops extracts the detected face co-ordinate value that is x, y, w, h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # (255, 0, 0), 2 = Blue color box and 2 thickness detected kelay nntr rectangle blue create houn display honr
        face_region = gray[y:y + h, x:x + w] # Extracting the face region that we convert already in grayscale , y : y+h → rows from top of face to bottom, x : x+w → columns from left of face to right, (Cut out only the face part from the grayscale image for comparison or saving.)

        # ---------- FIRST TIME REGISTRATION ----------
        if not os.path.exists(authorized_face_path):
            cv2.putText(frame, "Press S to Register Your Face", (20, 40), # (20, 40) co-ordinates on screen where to display
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)    # font-size, color,thickness

        # ---------- FACE MATCHING ----------
        else:
            authorized_face = cv2.imread(authorized_face_path, cv2.IMREAD_GRAYSCALE) # cv2.imread = read the img from the web cam, cv2.IMREAD_GRAYSCALE = read the img from web cam in grayscale , cv2.imread() this function always have 2 para 1 is path and another one is how to read the image (mode), Load the saved face image (authorized.jpg) in grayscale so we can compare it with the live face.
            diff = compare_faces(authorized_face, face_region)

            if diff < 500000:  # Threshold (simple)
                cv2.putText(frame, "ACCESS GRANTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                os.startfile(desktop_path)   # Opens Desktop Folder
                cap.release() # release the camera
                cv2.destroyAllWindows() # close the windows
                exit()
            else:
                cv2.putText(frame, "ACCESS DENIED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("Face Lock System", frame) # It is the title of the display window

    key = cv2.waitKey(1) & 0xFF

    # Save face (FIRST time)
    if key == ord('s') and not os.path.exists(authorized_face_path):
        cv2.imwrite(authorized_face_path, face_region) # detected first time face store in authorized.jpg file in grayscale
        print("Face Registered Successfully!")
        print("Run the program again to unlock.")
        break

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
