import cv2 as cv 
import matplotlib.pyplot as plt

img = cv.imread('./images/download.jpeg')

cv.imshow("Default Image", img)

#how to use matplotlib
plt.imshow(img)
plt.show()

#bgr to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

#bgr to hsv
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("HSV", hsv)

# bgr to lab
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("Lab", lab)

#bgr to rgb
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB", rgb)

plt.imshow(rgb)
plt.show()

#hsv to bgr
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow("hsv_bgr", hsv_bgr)

#lab to bgr
lab_bgr = cv.cvtColor(lab, cv.COLOR_Lab2BGR)
cv.imshow("lab_bgr", lab_bgr)


cv.waitKey(0)