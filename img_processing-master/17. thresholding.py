import cv2 as cv 

img = cv.imread("./images/istockphoto-483822556-170667a (1).jpg") #read in image
cv.imshow('Default', img)   # label it Cat

#graying our image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

#simple thresholding
threshold, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow("Simple Thresholded", thresh)

#thresh inverse
threshold, thresh_inv = cv.threshold(gray, 125, 255, cv.THRESH_BINARY_INV)
cv.imshow("Inverse Thresholded", thresh_inv)

#adaptive thresholding #computing on mean
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 5)
cv.imshow("Adaptive Thresholding Mean", adaptive_thresh)

#threshold using gausian
adaptive_thresh2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 5)
cv.imshow("Adaptive Thresholding Gaussian", adaptive_thresh2)





cv.waitKey(0) #until key is pressed