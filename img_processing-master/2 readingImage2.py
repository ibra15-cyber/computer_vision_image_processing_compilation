import cv2

img = cv2.imread("images/images.jpeg", 0)
print(type(img)) 
print(img) #will print the n array
print(img.shape) #pixels 1485, 990 is greater than my screen
print(img.ndim) #the dimension ie gray or rbg
cv2.imshow("Galaxy default", img)
cv2.waitKey(0)