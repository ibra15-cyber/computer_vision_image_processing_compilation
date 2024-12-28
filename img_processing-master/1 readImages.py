import cv2 as cv 

img = cv.imread("images/download.jpeg") #read in image
cv.imshow('Default', img)   # label it Cat
cv.waitKey(0) #until key is pressed