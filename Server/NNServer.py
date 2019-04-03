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

# Neural network variables
K.set_image_dim_ordering('tf')
# h5 filename constant
h5_fileName = "ASL_Alphabet_Tests/NN_test1.h5"


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
    print("Neural Network Error: %.2f%%" % (100-scores[1]*100))
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

# The image size which is accepted by the CNN
image_x, image_y = get_image_size()

print(image_y, image_x)

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

# single-threaded socket connection