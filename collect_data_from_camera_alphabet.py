# -*- coding: utf-8 -*-
# this code is not sufficient need to use some directory variable which is in the training alphabet file..so this code is not tested update : tested and ok
# and it was copied from training alphabet.py file to clear the coding space..
##################################################
#import h5py
#h5py.run_tests()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
import os
#print( cv.__version__ )

DIR = 'C:/Users/Pollen/project/dataset/new alphabet/D' #C:\Users\Pollen\project\dataset\new alphabet








#this part is for checking the camera and showing frames
a = 0 #to calculate the time in miliseconds
counter = 0

video=cv.VideoCapture(0)#create an object . Zero for external camera
while True:
    a = a + 1
#create a frame  object 
    check, frame = video.read()
    img_with_frame = cv.rectangle(frame, (160, 120), (480, 360), (0, 255, 0), 2)# (image, start point, end point , rectange color, thickness..frame with a green rectangle)
    cropped_image = img_with_frame[120:360, 160:480] #cropping image
    #print(check)#representing the matrix values
    #print(frame) #representing raw frame/image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  #converting to grayscale
    
    cv.imshow("capturing", img_with_frame) #shown the frame
    cv.imshow("cropped", cropped_image)
    cv.imshow("GRAY", gray)
    
    ##############################
    #this part is for collection of data ..training data..test data....
    if counter < 2000 : # 2000 images 
        cv.imwrite(os.path.join(DIR , ('%s.jpg' %counter)), cropped_image) # create and name with a number
        counter = counter + 1
    ##############################
    
    

    
    
    key = cv.waitKey(1)
    
    if key == ord(' '):# listening for space key press
        break
    
print(a)
#shutdown the camera
video.release()
cv.destroyAllWindows()