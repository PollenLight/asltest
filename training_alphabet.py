 # -*- coding: utf-8 -*-

#import h5py
#h5py.run_tests()
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
import os
#print( cv.__version__ )

DATADIR = 'C:/Users/Pollen/project/dataset/alphabet'  #change the last letter after finishing the data collection "C:/Users/Light/project/dataset/Y"#and also changed the name Light to Pollen after windows setup
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

imgh = 120 #img height
imgw = 160 #img width (horizontal)


# for category in CATEGORIES:#category is a random variable name
#     path = os.path.join(DATADIR, category) #paths to a,b,c,d......
#     for img in os.listdir(path):
#         img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap = "gray")
#         plt.show()
#         break
#     break#just ekta photo dekhay off korlo loop
# print(img_array.shape)#showing the resolution of the image




# resized_array = cv.resize(img_array, (imgw, imgh))# copied in userdefined function
# plt.imshow(resized_array, cmap = "gray")
# plt.show()





training_data = []

def create_training_data():
    for category in CATEGORIES:#category is a random variable name
        path = os.path.join(DATADIR, category) #paths to a,b,c,d......
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
                resized_array = cv.resize(img_array, (imgw, imgh))
                training_data.append([resized_array, class_num])
            except Exception as e:
                pass
  
            
  
create_training_data()
print(len(training_data)) # how many image&category are there in training data

import random
random.shuffle(training_data) #shuffle all data if not at first it sees all A then all B...not good.. so shuffle



# for sample in training_data:# only to see what data are stored in training data
#     print(sample[1])# printed the labels/a....z



#we dont need to split them up ..can use built in methods to properly do an out of sample test
X = [] # capital x is feature set
y = [] # lower case y is label..some times we can see trainx...test y..like so on

for features, label in training_data:
    X.append(features)# X has to be a numpy array
    y.append(label) # y can stay/be a list

X = np.array(X).reshape(-1, imgw, imgh, 1)
y = np.array(y)# i faced an error when reinstalled windows. then wrote this line from the link 
#( https://stackoverflow.com/questions/58682026/failed-to-find-data-adapter-that-can-handle-input-class-numpy-ndarray-cl)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

X = X / 255.0 #normalizing data to train models faster

model = Sequential()

model.add(    Conv2D(256, (2, 2), input_shape = X.shape[1:])    ) #64 - 128 has to check
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2))) #padding = same .suggested jodi problem hoy.

model.add(    Conv2D(256, (3, 3))    )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


#later added
model.add(    Conv2D(256, (4, 4))    )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(    Conv2D(64, (5, 5))    )
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())
model.add(Dense(512)) #64 -256
model.add(Activation("relu"))

model.add(Dense(512)) #64 -256#later added will delete
model.add(Activation("relu"))
# model.add(Dropout(0.2))#later added hasnt checked copied from a youtube video

model.add(Dense(24, activation = 'softmax')) #?????
#model.add(Activation('sigmoid'))  #softmax use korte hobe....in future


model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ['accuracy'])


model.fit(X, y, batch_size = 32, epochs = 4, validation_split = 0.3)# validation split is out of sample data age .1 chhilo..chang korlam




model.save('alphabet_training_test_manynodes.model')

# ------------------------------------------------------------------------------


# def prepare(filepath):
#     raw_img_array = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
#     resized_img_array = cv.resize(raw_img_array, (160, 120))
#     return resized_img_array.reshape(-1, 160, 120, 1)



# prediction = model.predict([prepare('test c.jpg')])
# print(prediction)


# prediction = model.predict([prepare('test l.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test m.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test o.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test t.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test u.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test v.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test w.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test x.jpg')])
# print(prediction)

# prediction = model.predict([prepare('test y.jpg')])
# print(prediction)







