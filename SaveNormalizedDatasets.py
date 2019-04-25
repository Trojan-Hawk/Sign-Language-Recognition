# Imports
import os
import numpy as np
import cv2
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
# import to access OS file system
import os.path as path
# GUI elements import
import tensorflow as tf
import socket
import blosc

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

# The image size which is accepted by the NN
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

print("Finished pickling loading step")

train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

print("Finished dataset pre-processing step")

# normalize the training and testing images
train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

print("Finished normalizing step")

# write the normalized objects to a files
with open('train_images_normalized', 'wb') as train_file:
    pickle.dump(train_images, train_file)
with open('test_images_normalized', 'wb') as test_file:
    pickle.dump(test_images, test_file)

print("Finished saving normalized objects to files")
