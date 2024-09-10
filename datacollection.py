import cv2

video = cv2.VideoCapture(1)

detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter your id: ")


count = 0


while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detection.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1

        face_region = gray[y:y + h, x:x + w]

        resized_face = cv2.resize(face_region, (100, 100))

        cv2.imwrite('Datasets/User.'+str(id)+"."+str(count)+".jpg", resized_face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 1)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

    if count > 100:
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection done .................")
