# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np

oldimage = np.zeros((240, 320), np.uint8)
movementimage = np.zeros((240, 320), np.uint8)
video=cv.VideoCapture(0)#from camera

while True:
    check, frame = video.read()#from camera
    
    
    rawimg =  cv.cvtColor(frame, cv.COLOR_BGR2GRAY)# raw img converted to grayscale img
    rawimg = cv.resize(rawimg, (320, 240))#imgwidth, imgheight (320, 240)
    #loop to compare prev and inew pixel(movement image)
    for i in range(240):
        for j in range(320):
            newpixel = rawimg[i,j]
            oldpixel = oldimage[i, j ]
            #print(abs(newpixel - oldpixel))
            
            if (newpixel != oldpixel): #abs(newpixel - oldpixel)>100 or newpixel != oldpixel
                movementimage[i, j] = 255
            else:
                movementimage[i, j] = 0
       
    #both = np.concatenate((rawimg, movementimage), axis = 1)
    #cv.imshow("capturingall", both)
    
    cv.imshow("raw", rawimg)
    cv.imshow("movement", movementimage)
    
    key = cv.waitKey(1)
    oldimage = rawimg
    
    
    if key == ord(' '):
        break
video.release()
cv.destroyAllWindows()