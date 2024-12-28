import cv2 as cv 
import numpy as np 

#create and show blank image
blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow("Blank", blank)

#change blank to green
blank[:] = 0,0,0 #0,0,255 red
cv.imshow("Blank", blank)

blank[200:300, 300:400] = 0,255,0 #color only certain portion.
cv.imshow("Green", blank)

#draw rectangle without a fill
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
cv.imshow("unfilled rectangle", blank)
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=cv.FILLED)
cv.imshow("Filled Rectangle", blank)

#square using shape fn instead of hardcodding
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,0,0), thickness=-1) #filled == -1
cv.imshow("Filled Rectangle", blank)


#draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (255,0,0), thickness=3) #or fill or -1
cv.imshow("Circle", blank)

# draw a line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=3) #or fill or -1
cv.imshow(" Green Line", blank)

cv.line(blank, (100,100), (300, 400), (0,255,0), thickness=3) #or fill or -1 blank.shape[1]//2, blank.shape[0]//2 hight and size
cv.imshow(" Green Line", blank)  

#5 write text
cv.putText(blank, "Hello my name is karim", (225, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), thickness=2)
cv.imshow("Text", blank)
cv.waitKey(0)

