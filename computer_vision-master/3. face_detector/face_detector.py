import cv2

#loading the face xml by haars
face_cascade=cv2.CascadeClassifier("./face_detector/haarcascade_frontalface_default.xml") #pass the right path

#loading image either the news or the photo
img=cv2.imread("./face_detector/news.jpg", 1) #we can choose to append 0 to make it appear gray

gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #but we are using this method for that

##you can choose to pass either the img or the gray image
faces=face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)
for x, y, w, h in faces:
    img=cv2.rectangle(img,(x,y),(x+w, x+h),(0, 255, 0),3) #image points and color and thickness

print(faces)

resized=cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

##reading the original file
cv2.imshow("ORIGINAL", img)
cv2.waitKey(0)
cv2.destroyAllWindows

##reading the grayed file
cv2.imshow("Edited", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##reading the grayed file
cv2.imshow("Resized", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

