# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np



video=cv.VideoCapture(0)#from camera

while True:
    check, frame = video.read()#from camera   
    cv.imshow("camera raw", frame)
    
    key = cv.waitKey(1)
    if key == ord(' '):
        break
video.release()
cv.destroyAllWindows()

#print (cv.__version__)
