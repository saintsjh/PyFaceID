import cv2
import serial

video = cv2.VideoCapture(1)

# sn = serial.Serial('COM3', 9600)

detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Jesse", "Random Baby", "Ashley", "Mom", "Ari"]


while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detection.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        faceRegion = gray[y:y + h, x:x + w]

        center_x = x + w // 2
        center_y = y + h // 2

        # sn.write(f'{center_x},{center_y}\n'.encode())

        resized_face = cv2.resize(faceRegion, (100, 100))

        serial, conf = recognizer.predict(resized_face)

        # Debugging prints
        print(f"Detected face with serial: {serial} and confidence: {conf}")

        if conf > 80:
            if 0 <= serial < len(name_list):
                cv2.putText(frame, name_list[serial], (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            else:
                cv2.putText(frame, "UNKNOWN", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 225), 1)
        else:
            cv2.putText(frame, "NEW PERSON", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 225), 1)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection done .................")
