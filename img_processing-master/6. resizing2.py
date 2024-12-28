import cv2 as cv

img = cv.imread("./images/people-ghana-accra-march-unidentified-ghanaian-boy-street-children-suffer-poverty-due-to-unstable-52131224.jpg")

cv.imshow("Image", img)

#instead of manually doing it straing forward
#we created a fn that we will pass images with para frame
#and hardcoded a frame
#int(frame.shape[0] * sacale ) = int(img.shape[0] * 0.75 )
def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)
    
    # resized_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    return cv.resize(frame, dimensions, interpoloaton=cv.INER_AREA) #same as above but we adding inter

cv.waitKey(0) #wait until zero is clicked to quite