# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


cap = cv.VideoCapture('test.mp4')#from video file

while True:
    ret, vframe = cap.read()#from video
    if ret == True:
        cv.imshow('videofileframe', vframe)
    
       
    #both = np.concatenate((rawimg, movementimage), axis = 1)
    #cv.imshow("capturingall", both)
    
    key = cv.waitKey(1)
    
    
    if key == ord(' '):
        break
cap.release()
cv.destroyAllWindows()