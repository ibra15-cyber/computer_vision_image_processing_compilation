import cv2
import time


# video=cv2.VideoCapture(0) #here int for webcame, more than 1 webcam means 0, 1, 2, etc.
video=cv2.VideoCapture("videos/c3ec2c55-40339c57.mp4") #for a video file

#track number of frame
fr=0
while True:
    fr+=1

    check, frame = video.read()
    # print(check) #my webcam isnt working
    # print(frame)
    # time.sleep() 
    #show captured frame
    frame1=cv2.imshow("Captured", frame)
    
    key=cv2.waitKey(1) #show the next frame in  1000 millisecond
    #incase you want to stop the frame make it zero
    #else make it any number you want it to keep changing
    #1 millisecond is the fastest
    
    #key is expecting a keybord
    #if 0 was passed it quits
    #any other numb here it goes to the next frame
    #because of our while loop
    #so we want to the q unicode value and tell it to break

    if key==ord('q'): #ord is very useful
        break

# print(fr)
video.release()
cv2.destroyAllWindows