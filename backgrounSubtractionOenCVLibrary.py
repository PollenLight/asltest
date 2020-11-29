# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import cv2 as cv


video=cv.VideoCapture(0)#from camera
		
fgbg = cv.createBackgroundSubtractorMOG2(history = 500, varThreshold = 160, detectShadows = 0)
#(history = 125, varThreshold = 100) #foregroundbackground
#detailed description official - https://docs.opencv.org/master/de/de1/group__video__motion.html

while True:
    check, frame = video.read()#from camera   
    fgmask = fgbg.apply(frame)
    cv.imshow("camera raw", frame)
    cv.imshow("camera subtract", fgmask)

    key = cv.waitKey(1)
    if key == ord(' '):
        break
video.release()
cv.destroyAllWindows()

#print (cv.__version__)