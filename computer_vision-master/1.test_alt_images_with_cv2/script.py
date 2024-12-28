import cv2

img = cv2.imread("./first/galaxy.jpg", 0)
print(type(img)) 
print(img) #will print the n array
print(img.shape) #pixels 1485, 990 is greater than my screen
print(img.ndim) #the dimension ie gray or rbg
cv2.imshow("Galaxy default", img)
cv2.waitKey(0)

# cv2.resize(imgname, shape)
resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
cv2.imshow("Galaxy Resized", resized_img) #show image with a caption
cv2.imwrite("./first/Galaxy_Resized.jpg", resized_img) #take the resized image and store it with the name
cv2.waitKey(0) #0 means until i decide to quit
cv2.destroyAllWindows()
