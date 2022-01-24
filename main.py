"""Antes de empezar, al autor del proyecto de IA creado por Carlos Mensias, debe aclarar que parte importante del código de este programa,
fué sacado de un video de la plataforma Youtube llamado "OpenCV Python TUTORIAL #4 for Face Recognition and Identification" y dicho video es perteneciente
al canal CodingEntrepeneurs
"""

import cv2
import pickle
import os





vid_cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
og_labels = {}

with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

counter = 0
while True:
    ret, frame = vid_cap.read()
    key = cv2.waitKey(1)



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor= 1.15,
                                          minNeighbors=2,
                                          minSize=(100,150),
                                          flags = cv2.CASCADE_SCALE_IMAGE,
                                          )

    for (x,y,w,h) in faces:
        roi_col = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        id_ , conf = recognizer.predict(roi_gray)

        if conf >= 47:
            name = labels[id_]
            cv2.putText(frame, name,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)



        cv2.rectangle(
            frame,
            (x,y),
            (x+w, y+h),
            (255,0,0),
            2
        )



    cv2.imshow("Reconocimiento facial", frame)



    if key == ord(" "):
        break
    elif key == ord("c"):
        counter += 1
        cv2.imwrite(f"image_{counter}.jpg", roi_col)


cv2.destroyAllWindows()
vid_cap.release()
