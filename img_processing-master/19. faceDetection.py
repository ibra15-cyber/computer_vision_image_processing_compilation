import cv2 as cv 

img = cv.imread("./images/istockphoto-1346125184-612x612.jpg") #read in image
cv.imshow('Default', img)   # label it Cat

#gray
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow("Gray", gray)

#HAAR CASSCAD
haar_cascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")

#adjust scaleFactor and minbours to reduce noise
faces_react = haar_cascade.detectMultiScale(gray, scaleFactor=1.65, minNeighbors=2)

#for faces in img
for (x,y,w,h) in faces_react:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("Detected faces", img)

print("Number of faces found =  {}".format(len(faces_react)))
print(f'Number of faces are: {len(faces_react)}' )

cv.waitKey(0) #until key is pressed