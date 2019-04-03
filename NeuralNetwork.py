import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing import image
# import to access OS file system
import os.path as path
# file dialog box and selector imports
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf


def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape


def get_num_of_classes():
	return len(os.listdir('gestures/'))


# defining the image shape
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

# other variables
K.set_image_dim_ordering('tf')
h5_filename = "ASL_Alphabet_Tests/NN_test1.h5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    # model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(num_of_classes, activation='sigmoid'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_of_classes, activation='softmax'))

    # sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    checkpoint1 = ModelCheckpoint(h5_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1]
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model, callbacks_list


def train():
    # build the model
    model, callbacks_list = cnn_model()
    # model.summary()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=30, batch_size=125, callbacks=callbacks_list)
    scores = model.evaluate(test_images, test_labels, verbose=0)
    print("Neural Network Error: %.2f%%" % (100-scores[1]*100))
    model.save(h5_filename)
    return model


def prediction(imageURL):
	prediction = ""
	test_image = image.load_img(imageURL, target_size=(image_x, image_y), color_mode="grayscale")
	test_image = np.array(test_image)
	test_image = np.reshape(test_image, (image_x, image_y, 1))
	test_image = np.expand_dims(test_image, axis=0)
	result = model.predict(test_image)
	print(result)


# check to see if the weights file exists
# if path.isfile(h5_filename):
#	print("Loading pre-compiled Neural Net Weights")
	# build the model
#	model, callbacks_list = cnn_model()
#	model.load_weights(h5_filename)
# else:
print("Creating Neural Net and saving Weights to file")
# train and fit the model
model = train()

# Final evaluation of the model
scores = model.evaluate(test_images, test_labels, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# loop until the program is exited
while True:
	# open a new file explorer window
	root = tk.Tk()
	root.withdraw()

	try:
		# store the url of the selected file
		file_path = filedialog.askopenfilename()
		# call the prediction method of our model
		prediction(file_path)
	except():
		print("No file selected. Exiting application!")
		break
