import cv2 as cv 
import numpy as np

img = cv.imread("images/portrait-of-an-arab-man-BRMRWC.jpg") #read in image
cv.imshow('Default', img)   # label it Cat

blank = np.zeros(img.shape[:2], dtype="uint8")
cv.imshow("Blank image", blank)

# circle_img = cv.circle(blank.copy(), (img.shape[1]//2 - 290, img.shape[0]//2 -100), 100, (223), -1  )

# rect_img = cv.rectangle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), (img.shape[1]//2 - 600, img.shape[0]//2 -300 ), (223), -1  )


# create a circle using blank above and make th size half of black with radius 100 color whiet and thikness -1
mask_circle = cv.circle(blank.copy(), (img.shape[1]//2 - 290, img.shape[0]//2 -100), 100, (223), -1  )
cv.imshow("Masked", mask_circle)

mask_rect = cv.rectangle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), (img.shape[1]//2 - 600, img.shape[0]//2 -300 ), (223), -1  )
cv.imshow("Masked", mask_rect)

# wired_shape = cv.bitwise_and(circle_img, rect_img)
# cv.imshow("Wired shap", wired_shape)

# using images in a form of circle and rectangle
# mask_circle2 = cv.circle(circle_img, (img.shape[1]//2 - 290, img.shape[0]//2 -100), 100, (223), -1  )
# cv.imshow("Masked", mask_circle)

# mask_rect2 = cv.rectangle(rect_img, (img.shape[1]//2, img.shape[0]//2), (img.shape[1]//2 - 600, img.shape[0]//2 -300 ), (223), -1  )
# cv.imshow("Masked", mask_rect)

# without the copy() the file will first have a circle and then a rectangle

# merge our image with the same image and mask with any of the masked above circle or rect
masked = cv.bitwise_and(img, img, mask=mask_circle)
# masked2 = cv.bitwise_and(circle_img, rect_img, mask=mask_circle2)
cv.imshow("Masked", masked)
# cv.imshow("Masked2", masked2)
cv.waitKey(0) #until key is pressed