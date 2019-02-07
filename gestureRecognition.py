# Imports
import cv2, os
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
# import to access OS file system
import os.path as path
from PIL import Image
from resizeimage import resizeimage

K.set_image_dim_ordering('tf')
cap = cv2.VideoCapture(0)
# file name constant
h5FileName = "test2.h5"


# returns the number of classes of gestures
def get_num_of_classes():
    return len(os.listdir('gestures/'))


# Getting the accepted image size values
def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)
    return img.shape


# defines the model that will be used
def model_dfn():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    # sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    checkpoint1 = ModelCheckpoint(h5_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1]
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model, callbacks_list


# predicts the gesture based on the pre-trained model
def prediction(test_image):
    # resize the image to the accepted values
    test_image = cv2.resize(mask, (image_x, image_y), interpolation=cv2.INTER_AREA)
    # create an array from the image
    test_image = np.array(test_image)
    # expand the dimensions of the array to match the CNN accepted input
    # # test_image = np.expand_dims(test_image, axis=0)
    # # test_image = np.expand_dims(test_image, axis=3)
    test_image = np.reshape(test_image, (image_x, image_y, 1))
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    print("RESULT: ", result)


# The image size which is accepted by the CNN
image_x, image_y = get_image_size()
# build the model shell
model = model_dfn()

# check to see if the weights file exists
if path.isfile(h5FileName):
    print("Loading pre-compiled CNN Weights")
    model.load_weights(h5FileName)
else:
    print("Training CNN!")


while(1):
        
    # try to display data to the user
    try:  
        # get a handle on the display
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)
        
        # define region of interest(Green Box)
        roi = frame[100:320, 100:320]

        cv2.rectangle(frame, (100, 100), (320, 320), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([0, 38, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin colour image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=2)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # Output of the visual data input (Camera) *************************************************  TESTING!
        # print(cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find contour of max area(hand)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        
        # skew the contours a little
        epsilon = 0.0005*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # make a convex bounds around hand
        hull = cv2.convexHull(cnt)
        
        # define area of bounds and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
        # find the percentage of area not covered by hand in bounds
        arearatio = ((areahull-areacnt)/areacnt)*100
    
        # find the defects in bounds with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        # l = no. of defects
        defectCount = 0
        
        # finding no. of defects due to finger spacing
        # and drawing lines around the captured image
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            # distance between point and convex hull
            d = (2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                defectCount += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)
            
            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        # Make a prediction based on the mask image extracted
        prediction(mask)

        # display the black/white mask extracted from the area of interest
        cv2.imshow('mask', mask)
        # display the camera input data
        cv2.imshow('frame', frame)
    except:
        print("Cannot access camera or none exists!")
        pass
        
    # If the user has pressed the escape key, exit the loop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
# On exit, remove all windows created
cv2.destroyAllWindows()
cap.release()
