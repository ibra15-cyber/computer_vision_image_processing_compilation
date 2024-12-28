import cv2 as cv 

img = cv.imread("images/portrait-of-an-arab-man-BRMRWC.jpg") #read in image
cv.imshow('Default', img)   # label it Cat

# averaging
average = cv.blur(img, (3,3))
cv.imshow("Average Blur", average)

#gausian blur
gauss = cv.GaussianBlur(img, (7,7), 0)
cv.imshow("Gausian Blur", gauss)

# Median Blur 
median = cv.medianBlur(img, 3)
cv.imshow("Median Blur", median)

# Bilateral
bilateral = cv.bilateralFilter(img, 5, 35, 25)
cv.imshow("Bilateral", bilateral)




cv.waitKey(0) #until key is pressed