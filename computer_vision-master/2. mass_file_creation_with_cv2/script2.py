import cv2 #for controlling and writing images
import glob #for finding path name

images=glob.glob("./second/*.jpg") #grab anything with jpg

for image in images:
    img=cv2.imread(image, 0) #read the image in black and white
    re=cv2.resize(img, (100, 100)) #resize the image
    cv2.imshow("Hey", re) #show the the reized images with a hey as caption
    cv2.waitKey(0) #wait for 500mills
    cv2.destroyAllWindows() #then destroy
    cv2.imwrite("./second/resized_"+image, re) #write the resize image, re as resized + default image name as gotten by glob

