# -*- coding: utf-8 -*-
#will update this from the random testing file

#import h5py
#h5py.run_tests()
import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv
import os

#i = 0 # created to count all the images in 'words' folder


DATADIR = 'C:/Users/Pollen/project/dataset/words'  #change the last letter after finishing the data collection "C:/Users/Light/project/dataset/Y"#and also changed the name Light to Pollen after windows setup
CATEGORIES = ["ADULT", "AGAIN", "AIRPLANE", "BABY", "BAD", \
               "BEAUTIFUL", "BED", "BETTER", "BOOK", "BROKE", \
               "CAR", "DAY", "DEAF", "DEATH", "DOCTOR", \
               "DONT KNOW", "FATHER", "FEAR", "FOOD", \
               "GLASS", "GOOD", "GOOD BYE", "GUILT", "HELLO", \
               "HELP", "INFECTION", "INHALER", "LEARN", "LIFE", "LOVE", \
               "MOON", "MOTHER", "NEVER", "NO", "PAIN", \
               "PERSON", "PHONE", "PICTURE", "PLEASE", "SHOES", \
               "SHOWER", "SMILE", "THANK YOU", \
               "TRUST", "UNDERSTAND", "WAKE UP", "WHAT", "WHEN", \
               "WORK", "YES"]

#***********************this part is for checking if the program can access the images************************
#for category in CATEGORIES:#category is a random variable name
#    path = os.path.join(DATADIR, category) #paths to words 1by1, concatenating the paths.
#    for img in os.listdir(path):
#        #i=i+1 #for counting the images
#        #print(i) #for counting the images
#        img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
#        plt.imshow(img_array, cmap = "gray")
#        plt.show()
#        break
#    break
#***********************this part is for checking if the program can access the images************************


#*******************************checking the resized images*******************************************

#print(img_array.shape)# to show the resolution and colour of the img_array variable
imgh = 360    #360 #480 #240 #img height
imgw = 480    #480#640 #320 #img width (horizontal)
#resized_array = cv.resize(img_array, (imgw, imgh))# this line is copied in userdefined function
#plt.imshow(resized_array, cmap = "gray")
#plt.show()


#*******************************checking the resized images*******************************************


##print(type(CATEGORIES)) #CHECKING WHICH TYPE OF DATA TYPE THE 'categories' is =  list


training_data = []

def create_training_data():
    for category in CATEGORIES:#category is a random variable name
        path = os.path.join(DATADIR, category) #paths to words 1by1, concatenating the paths.
        class_num = CATEGORIES.index(category)#JA BUJHLAM .FIRST CATEGORY KE 1 NUMBER E RAKHBE CLASS_NUM AND SO ON
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                resized_array = cv.resize(img_array, (imgw, imgh))
                training_data.append([resized_array, class_num])
            except Exception as e:
                pass
   
print(type(training_data))         
            
            
create_training_data() # calling my defined function above
print(len(training_data)) # how many image&category are there in training data

import random
random.shuffle(training_data) #shuffle all data if not at first it sees all A then all B...not good.. so shuffle


for sample in training_data:# only to see what data are stored in training data
    print(sample[1])# printed the labels/a....z


#we dont need to split them up ..can use built in methods to properly do an out of sample test
X = [] # capital x is feature set
y = [] # lower case y is label..some times we can see trainx...test y..like so on

for features, label in training_data:
    X.append(features)# X has to be a numpy array
    y.append(label) # y can stay/be a list

X = np.array(X).reshape(-1, imgw, imgh, 1) #https://www.w3schools.com/python/numpy_array_reshape.asp




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

X = X / 255.0 #normalizing data to train models faster

model = Sequential()
model.add(    Conv2D(64, (3, 3), input_shape = X.shape[1:])    )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(    Conv2D(64, (3, 3))    )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(50)) #????? afaik the number of categories i want to classify
model.add(Activation('sigmoid'))


model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ['accuracy'])


model.fit(X, y, batch_size = 32, epochs = 1, validation_split = 0.1)# validation split is out of sample data



#---------------------------prediction testing---------------https://youtu.be/A4K6D_gx2Iw?t=694_________
#import tensorflow as tf
#
#def prepare(filepath):
#    raw_img_array = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
#    resized_img_array = cv.resize(raw_img_array, (480, 360))
#    return resized_img_array.reshape(-1, 480, 360, 1)




# def prepare(filepath):
#     raw_img_array = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
#     resized_img_array = cv.resize(raw_img_array, (160, 120))
#     return resized_img_array.reshape(-1, 160, 120, 1)



# prediction = model.predict([prepare('test y.jpg')])
# print(prediction)

#print(CATEGORIES[int(prediction[0][0])])








#import cv2 as cv
#raw_img_array = cv.imread('test goodbye.jpg', cv.IMREAD_GRAYSCALE)
#cv.imshow('window', raw_img_array)
#
#cv.waitKey(0) #waits for user to press any key -necessary to avoid Python kernel form crashing)
#cv.destroyAllWindows() 
#-----------------------------------------------------------------------------------------