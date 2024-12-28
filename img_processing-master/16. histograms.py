import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np 

img = cv.imread("images/istockphoto-483822556-170667a (1).jpg") #read in image
cv.imshow('Default', img)   # label it Cat

blank = np.zeros(img.shape[:2], dtype="uint8")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

#cut out a circle in blank with white
circle  = cv.circle(blank, (img.shape[1]//2 - 97, img.shape[0]//2 -50), 100, 255, -1)
cv.imshow("Circle", circle)
 
#merge gray and gray with the circle showing only the cirlce part
mask_gray = cv.bitwise_and(gray, gray, mask=circle)
cv.imshow("Mask_Gray", mask_gray)

mask_rgb = cv.bitwise_and(img,img, mask=circle)
cv.imshow("Mask RGB", mask_rgb)

# gray histogram
gray_list = cv.calcHist([gray], [0], mask_gray, [256], [0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(gray_list)
plt.show()

# color histogram
plt.figure()
plt.title("Colored Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256]) 
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0) #until key is pressed



###passign mask_rgb to in line 41 causes error