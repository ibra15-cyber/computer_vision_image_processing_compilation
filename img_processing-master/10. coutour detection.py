import cv2 as cv 
import numpy as np

#importing image and reading it
img = cv.imread("./images/istockphoto-483822556-170667a (1).jpg")
cv.imshow("Default", img)

#setting a blank page with numpy and displaying it
blank = np.zeros(img.shape, dtype='uint8')
cv.imshow("Blank", blank)

#converting our image to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

#bluring the gray image
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#to find edges with canny
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Edges img", canny)

#finding edges on the blured images
canny_blur = cv.Canny(blur, 125, 175)
cv.imshow("Canny_blur Edges", canny_blur)

#threshold can replace canny
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Threshold", thresh)

#getting contours with destructuring 
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #returns 2 variable retr very important
print(f'{len(contours)} contour(s) found!')

#drawing contours on blank page
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow("Contours Drawn", blank)



cv.waitKey(0)