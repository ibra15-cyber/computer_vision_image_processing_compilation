import cv2

img = cv2.imread('images/images.jpeg')

print(img.ndim) #shows if image is gray or rbg

#reducing image size by half
resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

#sshow resized image
cv2.imshow("Resized Image", resized_img)
cv2.waitKey(0)