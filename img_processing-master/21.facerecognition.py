import numpy as np 
import cv2 as cv 

people = ['obama', 'mandela', 'trump']

#for detecting images
haar_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

#loading features and labels
# features = np.load('./features.npy', allow_pickle=True)
# labels = np.load('./labels.npy')


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('./face_trained.yml')


#passing our image to detect to be see if it can be detected
img = cv.imread(r'.\train\obama\images.jpg')

#converting image to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

faces_react = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces_react:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]},with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("Detected", img)

cv.waitKey(0)
