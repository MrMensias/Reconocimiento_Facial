"""Antes de empezar, al autor del proyecto de IA creado por Carlos Mensias, debe aclarar que parte importante del código de este programa,
fué sacado de un video de la plataforma Youtube llamado "OpenCV Python TUTORIAL #4 for Face Recognition and Identification" y dicho video es perteneciente
al canal CodingEntrepeneurs
"""


import numpy as np
import os
from PIL import Image
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, "Images")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(img_dir):
    for file in files:

        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            size = (100,100)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array,
                                                  scaleFactor=1.15,
                                                  minSize=(100,100),
                                                  minNeighbors=3,
                                                  flags = cv2.CASCADE_SCALE_IMAGE)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")



