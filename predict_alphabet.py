

import cv2
import tensorflow as tf

model = tf.keras.models.load_model('alphabet_training_test.model')



#DIR = 'C:/Users/Pollen/project/@TEST AREA'


video=cv2.VideoCapture(0)#create an object . Zero for external camera
while True:
    check, frame = video.read()
    img_with_box = cv2.rectangle(frame, (160, 120), (480, 360), (0, 255, 0), 2)# (image, start point, end point , rectange color, thickness..frame with a green rectangle)
    cv2.imshow("capturing", img_with_box)
    
    cropped_image = img_with_box[120:360, 160:480] 
    # cv2.imshow("cropped", cropped_image)
    
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)  #converting to grayscale
    # cv2.imshow("gray", gray)
    
    
    resized_img_array = cv2.resize(gray, (160, 120))
    cv2.imshow("window", resized_img_array)
    flatten_image =resized_img_array.reshape(-1, 160, 120, 1)
    
    prediction = model.predict([flatten_image])
    print(prediction)
      
    key = cv2.waitKey(1)
    if key == ord(' '):# listening for space key press
        break
    
video.release()
cv2.destroyAllWindows()




