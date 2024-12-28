import cv2 as cv
import numpy as np

img = cv.imread('./images/download.jpeg')
cv.imshow("Default", img)

#translation
def translate (img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

#-x > left
#-y > up
#x > right
#y > down 

translated = translate(img, 50, 50)
cv.imshow("Translated", translated)

#rotation
def rotate (img, angle, rotationPoint=None):
    (height, width) = img.shape[:2] #get height and width through destructuing

    if rotationPoint is None:
        rotationPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow("Rotation", rotated)

rotated_rotated = rotate(rotated, -45)  #you can send it back to the original
cv.imshow("Rotated Rotated", rotated_rotated)


#resizing 
reized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", reized)

#flipping
flip = cv.flip(img, -1) #0 vertical along x, 1 horizontal over y; -1 vertical and horizontal
cv.imshow("Flipped",flip)

#cropping
cropped = img[20:100, 50:300]
cv.imshow("Cropped", cropped)

cv.waitKey(0)