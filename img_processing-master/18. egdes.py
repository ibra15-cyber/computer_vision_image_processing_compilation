import cv2 as cv 
import numpy as np 

img = cv.imread("./images/download.jpeg") #read in image
cv.imshow('Default', img)   # label it Cat

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", gray)

#laptacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian", lap)

#Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combine_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow("Combined", combine_sobel)

cv.imshow("SobelX", sobelx)
cv.imshow("SobelY", sobely)

#canny
canny = cv.Canny(gray, 150, 175)
cv.imshow("Canny", canny)














cv.waitKey(0) #until key is pressed