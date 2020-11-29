import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv
import time
import os
import sys
 


DATADIR = 'C:/Users/Pollen/project/dataset/MOON/'     #change the last letter after finishing the data collection "C:/Users/Light/project/dataset/Y"#and also changed the name Light to Pollen after windows setup


while True:
  
    old_image = np.zeros((480, 640), np.uint8)  
    new_image = np.zeros((480, 640), np.uint8) 

    video=cv.VideoCapture(0)#from camera
		 
    fgbg = cv.createBackgroundSubtractorMOG2(history = 500, varThreshold = 160, detectShadows = 0)
    #(history = 125, varThreshold = 100) / (history = 500, varThreshold = 160, detectShadows = 0) #foregroundbackground
    #detailed description official - http s://docs.opencv.org/master/de/de1/group__video__motion.html
    start = time.time()#https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
    
    while True: 
        check, frame = video.read()#from camera   
        fgmask = fgbg.apply(frame)
        cv.imshow("camera raw", frame)
        cv.imshow("camera subtract", fgmask)
         
        
        end = time.time()
        #print(end-start) # showing the seconds as they pass by
        if((end-start)>1):#5 second por theke window open hobe and frame dekhano &adding start korbe
            new_image = cv.add(old_image, fgmask)
            cv.imshow("added images", new_image)  
            old_image = new_image
            
        if((end-start)>2):
            cv.imwrite(os.path.join(DATADIR , ('%s.jpg'%int(time.time()))), old_image)#time value is float so converted to int #https://www.programiz.com/python-programming/time
            break
            
            
            
        key = cv.waitKey(1)# sleeps for X miliseconds, waiting for any key to be pressed.
        if key == ord(' '):
            video.release()
            cv.destroyAllWindows()
            sys.exit(0) #0 used for meaning..succesfull t ermination.
        
    video.release()
    cv.destroyAllWindows()