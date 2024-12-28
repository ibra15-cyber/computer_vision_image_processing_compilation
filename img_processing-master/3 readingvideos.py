import cv2 as cv 

capture = cv.VideoCapture('videos/c3ec2c55-40339c57.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()

cv.destroyAllWindows()