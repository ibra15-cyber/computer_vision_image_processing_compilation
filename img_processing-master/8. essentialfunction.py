import cv2 as cv 
img = cv.imread("./images/images.jpeg")
cv.imshow("default", img)
#converting to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img_hsv = color.rgb2hsv(img)  #using skimage
cv.imshow("Gray", gray)

#blur
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
cv.imshow("blur", blur)

#edge cacade finding edges
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Edges", canny)

canny = cv.Canny(blur, 125, 175)
cv.imshow("Canny with blur", canny)

#dilating image
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow("Dilated", dilated)

#erroding of edges ie getting back edges
eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow("Eroded", eroded)

# resize and crop
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC ) #interpolation has diff option to meke it look nicer
cv.imshow("Resized", resized)

#cropping or chopping
croped = img[20:100, 50:100]
cv.imshow("Cropped", croped)

cv.waitKey(0)