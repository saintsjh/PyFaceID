import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

path = "Datasets"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePaths in imagePath:
        faceImage = Image.open(imagePaths).convert('L')
        faceNP = np.array(faceImage, 'uint8')
        ID = int(os.path.split(imagePaths)[-1].split(".")[1])

        # Detect faces in the image
        faces_detected = detector.detectMultiScale(faceNP)

        for (x, y, w, h) in faces_detected:
            face = faceNP[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (100, 100))
            faces.append(resized_face)
            ids.append(ID)
            cv2.imshow("Training", resized_face)
            cv2.waitKey(1)
    return ids, faces
    #     ID = int(ID)
    #     faces.append(faceNP)
    #     ids.append(ID)
    #     cv2.imshow("Training", faceNP)
    #     cv2.waitKey(1)
    # return ids, faces


ids, facedata = getImageID(path)
ids = np.array(ids)
recognizer.train(facedata, ids)
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training complete..................")
