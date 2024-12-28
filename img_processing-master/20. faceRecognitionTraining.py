import os
import cv2 as cv
import numpy as np 

# created a list and loop through a folder each time appending item in our list
# p = []
# for i in os.listdir(r'D:\School\400\DCIT 407 IMAGE PROCESSING\mytutorial\freecodecamp_image_processing\train'):
#     p.append(i)
# print(p) #lists our items


people = ['obama', 'mandela', 'trump']


DIR = r'./train'

haar_cascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        # print(path)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            # print(img_array)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_react = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_react:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training done..................................")

# #debugging
print(f'Length of the features = {len(features)}')
print(f'Length of the features = {len(labels)}')

# #covernting to numpy array
features = np.array(features, dtype='object')
labels = np.array(labels)

# print(features)

face_recognizer = cv.face.LBPHFaceRecognizer_create() #instantiate face recognizer
print(face_recognizer)

#train the recognizer on the features list and the label list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml') #save the trained file
np.save('features.npy', features)
np.save('labels.npy', labels)