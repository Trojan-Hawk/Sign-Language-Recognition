# Imports
import cv2, os
import numpy as np
import math
import pickle
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
# GUI elements import
from tkinter import *
import tensorflow as tf

# external XML file which defines the cascade face detection classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# number of hsv colour samples to be extracted
max_samples = 15
# define the default range of skin color
lower_skin = np.array([0, 0, 0], dtype=np.uint8)
upper_skin = np.array([255, 255, 255], dtype=np.uint8)
# Difference of 220 between start and end gives an adequate region of interest
regionStartX = 250
regionStartY = 130
regionEndX = 470
regionEndY = 350
# global counter
global count
count = 0
# boolean control
skinToneOK = False
# skin tone list
skinToneSamples = []

# Neural network variables
K.set_image_dim_ordering('tf')
# h5 filename constant
h5_fileName = "ASL_Alphabet_Tests/NN_test1.h5"
# openCV video capture
cap = cv2.VideoCapture(0)


# --------------------------------------------- Neural Network Methods -------------------------------------------------
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
    global model
    model = Sequential()
    model.add(Flatten(input_shape=(image_x, image_y, 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))
    # sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    checkpoint1 = ModelCheckpoint(h5_fileName, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1]
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model, callbacks_list


def train():
    # model.summary()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=5, batch_size=300, callbacks=callbacks_list)
    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    model.save(h5_fileName)


# predicts the gesture based on the pre-trained model
def prediction(test_image):
    global label_encoder
    # resize the image to the accepted values
    test_image = cv2.resize(test_image, (image_x, image_y), interpolation=cv2.INTER_AREA)
    # create an array from the image
    test_image = np.array(test_image)
    # expand the dimensions of the array to match the CNN accepted input
    # # test_image = np.expand_dims(test_image, axis=0)
    # # test_image = np.expand_dims(test_image, axis=3)
    test_image = np.reshape(test_image, (image_x, image_y, 1))
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    # print("RESULT: ", result)
    inverted = label_encoder.inverse_transform([argmax(result)])

    return inverted[0]


# --------------------------------------------- Input data methods -----------------------------------------------------
# method that takes in a region of interest and extracts samples of skin tone(HSV)
def sample_skin_tones(roi):
    # convert to HSV (Hue-Saturation-Value)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # display the HSV mask extracted from the area of interest
    # cv2.imshow('FaceHSV', hsv)
    # use colour quantization to extract the most common colours
    Z = roi.reshape((-1, 3))
    # convert to numpy float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans() clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back into uint8
    center = np.uint8(center)
    # flatten the pixles
    face1DArray = center[label.flatten()]
    # restore the original image shape
    face2DArray = face1DArray.reshape((roi.shape))

    # destroy the previous window
    cv2.destroyWindow('Quantization')
    # display the extracted face
    cv2.imshow('Quantization', face2DArray)

    # find the mid-point of the array
    height, width, vals = face2DArray.shape
    # cut the height and width in half and round the values(ensure int values)
    height = int(round(height / 2))
    width = int(round(width / 2))
    # store the hsv value at the center point
    skinToneSamples.append(face2DArray[height][width])
    # store the hsv value at the top half center point
    height1 = round(height * 0.8)
    skinToneSamples.append(face2DArray[height1][width])
    # store the hsv value at the bottom half center point
    height2 = round(height * 1.2)
    skinToneSamples.append(face2DArray[height2][width])


# method that uses openCV's cascade classifier functionality
def facial_recognition():
    try:
        # get a handle on the display
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        # convert the colour range of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # using openCV's cascade face detection from external xml file
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # Use this region to create a view of the extrapolated face
            roi = frame[y:y + h, x:x + w]
            # Show the area of the rectangle (Red Box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if roi is not None:
                # calls the method to sample skin tone
                sample_skin_tones(roi)
                return True
    except:
        print("Cannot find face!")
        return False


# define the skin tone range by selecting samples from the list
def define_skin_tone_range():
    # degree of skew
    skew_degree = 2.2
    # link to global variables
    global lower_skin, upper_skin
    # calculate the first and last quarter sample selections and ensure int val
    lower = int(round((max_samples / 100) * 30))
    upper = int(round((max_samples / 100) * 70))
    # if we have reached the max samples
    # sort the list of hsv values
    sorted_samples = sorted(skinToneSamples, key=lambda x: x[0])
    # take the value at position 50, which will be the lower skin tone range
    lower_skin = np.array(sorted_samples[lower-1:lower], dtype=np.uint8)
    # take the value at position 150, which will be the higher skin tone range
    upper_skin = np.array(sorted_samples[upper-1:upper], dtype=np.uint8)

    # pre-set variables to define the range accepted, range further defined by extracted skin tone
    new_lower = np.array((0, lower_skin[0][1]/skew_degree, lower_skin[0][2]/skew_degree), dtype=np.uint8)
    lower_skin = new_lower
    new_upper = np.array((upper_skin[0][0]/skew_degree, 255, 255), dtype=np.uint8)
    upper_skin = new_upper


# ---------------------------------------------- User Interface --------------------------------------------------------
# font style and size variables
font = cv2.FONT_HERSHEY_PLAIN
fontScale = 2


def close_gui(GUI):
    # close the GUI
    GUI.destroy()


def text_prompt(path, title):
    # create the gui window
    GUI = Tk()
    # Open the file with the text contents to use
    instructions = os.open(path, os.O_RDWR)
    # title
    GUI.title(title)
    # text display
    text = Text(GUI, font=(font, 13), padx=25, pady=25, width=65, height=12, wrap=WORD)
    # inserting the text
    text.insert(INSERT, os.read(instructions, 510))
    # prevent the user from editing the text
    text.config(state=DISABLED)
    # pack the text
    text.pack()
    # close the file
    os.close(instructions)
    # add a button to the gui window
    button = Button(GUI, text="OK", command=lambda: close_gui(GUI), padx=30, pady=5, bg="cornflower blue")
    button.place(relx=.5, rely=1, x=2, y=-10, anchor=S)
    # GUI event loop
    GUI.mainloop()


# def prediction_display()


# --------------------------------------------------- Main -------------------------------------------------------------

# The image size which is accepted by the CNN
image_x, image_y = get_image_size()

# testing and training data preparation
with open("train_images", "rb") as f:
    train_images = np.array(pickle.load(f))
with open("train_labels", "rb") as f:
    train_labels = np.array(pickle.load(f), dtype=np.int32)

with open("test_images", "rb") as f:
    test_images = np.array(pickle.load(f))
with open("test_labels", "rb") as f:
    test_labels = np.array(pickle.load(f), dtype=np.int32)
train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

# normalize the training and testing images
train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)



from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
data = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
values = array(data)
# print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded[0])
# invert first example



# build the model shell
model, callbacks_list = model_dfn()

# check to see if the weights file exists
if path.isfile(h5_fileName):
    print("Loading pre-compiled CNN Weights")
    # load the pre-compiled weights
    model.load_weights(h5_fileName)
    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
else:
    print("No weights file '" + h5_fileName + "' exists.")
    train()

# prompt the user using the init text file resource
text_prompt("text_resources/init.txt", "Initialize Facial Recognition") ################################

# infinite while loop
while(1):
    # try to display data to the user
    try:
        # if the skin tone range has not been defined yet
        if not skinToneOK:
            hsvExtracted = facial_recognition()
            # if a skin tone has been added
            if hsvExtracted:
                # increment and print the counter
                count += 1
                print(count)

            if count >= max_samples:
                # call the skin tone range method
                define_skin_tone_range()
                # destroy the facial recognition window
                cv2.destroyWindow('Quantization')
                # set the boolean control to true
                skinToneOK = True
        else:
            # get a handle on the display
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            kernel = np.ones((3, 3), np.uint8)

            # define region of interest(Green Box)
            roi = frame[regionStartX:regionEndX, regionStartY:regionEndY]

            cv2.rectangle(frame, (regionStartY, regionStartX), (regionEndY, regionEndX), (0, 255, 0), 0)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # skin colour range
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # if the values are incorrect for the skin tone
            if mask is None:
                print("Mask has no value.")

            # extrapolate the hand to fill dark spots within
            mask = cv2.dilate(mask, kernel, iterations=2)

            # blur the image
            mask = cv2.GaussianBlur(mask, (5, 5), 100)

            # Output of the visual data input (Camera)
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

            # if there are defects
            if defects is not None:
                # l = no. of defects
                defectCount = 0

                # code for finding no. of defects due to fingers
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

                print(defectCount)

                # make a guess
                displayText = "The sign is: " + prediction(mask)
                # add the text to the frame
                cv2.putText(frame, displayText, (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)

                # display the black/white mask extracted from the area of interest
                cv2.imshow('mask', mask)
                # display the camera input data
                cv2.imshow('frame', frame)
    except ValueError:
        # inform the user that no sign is detected
        displayText = "No sign detected."
        # add the text to the frame
        cv2.putText(frame, displayText, (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
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
