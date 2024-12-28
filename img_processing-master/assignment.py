import cv2
import numpy as np
#cv2 is used for OpenCV library

image = cv2.imread("./images/download.jpeg")
#imread is use to read an image from a location

img = image.astype(np.float64)/255.
K = 1 - np.max(img, axis=2)
C = (1-img[...,2] - K)/(1-K)
M = (1-img[...,1] - K)/(1-K)
Y = (1-img[...,0] - K)/(1-K)

CMYK_image= (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)


cv2.imshow("Original Image", image)
cv2.imshow("CMYK Image", CMYK_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load image
# bgr = cv2.imread('images/download.jpeg')

# # Make float and divide by 255 to give BGRdash
# bgrdash = bgr.astype(np.float64)/255.

# # Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
# K = 1 - np.max(bgrdash, axis=2)

# # Calculate C
# C = (1-bgrdash[...,2] - K)/(1-K)

# # Calculate M
# M = (1-bgrdash[...,1] - K)/(1-K)

# # Calculate Y
# Y = (1-bgrdash[...,0] - K)/(1-K)

# # Combine 4 channels into single image and re-scale back up to uint8
# CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
