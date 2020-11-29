# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time


old_image = np.zeros((480, 640), np.uint8)
new_image = np.zeros((480, 640), np.uint8) 



video=cv.VideoCapture(0)#from camera
		
fgbg = cv.createBackgroundSubtractorMOG2(history = 100, varThreshold = 60, detectShadows = 0)
#(history = 125, varThreshold = 100) #foregroundbackground
#detailed description official - https://docs.opencv.org/master/de/de1/group__video__motion.html
start = time.time()#https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
while True: 
    check, frame = video.read()#from camera   
    fgmask = fgbg.apply(frame)
    cv.imshow("camera raw", frame)
    cv.imshow("camera subtract", fgmask)
    
    
    end = time.time()
    print(end-start)
    if((end-start)>2):#5 second por theke window open hobe and frame dekhano &adding start korbe
        new_image = cv.add(old_image, fgmask)
        cv.imshow("added images", new_image)
    
    old_image = new_image
    
    
    key = cv.waitKey(1)
    if key == ord(' '):
        break
video.release()
cv.destroyAllWindows()

#print (cv.__version__)