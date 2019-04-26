import cv2
import numpy as np
from array import *

# external XML file which defines the cascade face detection classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# method that uses openCV's cascade classifier functionality
def facial_recognition(frame):
    try:
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
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if roi is not None:
                # calls the method to sample skin tones and returns the values
                return sample_skin_tones(roi)
            else:
                print("No face detected")
                return None
    except():
        print("Error")
        return None


# method that takes in a region of interest and extracts samples of skin tone(HSV)
def sample_skin_tones(roi):
    # convert to numpy float32
    Z = np.float32(roi)
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

    # find the mid-point of the array
    height, width, vals = face2DArray.shape
    # cut the height and width in half and round the values(ensure int values)
    height = int(round(height / 2))
    width = int(round(width / 2))

    # define the list
    samples = []

    # store the hsv value at the center point
    samples.append(face2DArray[height][width])
    # store the hsv value at the top half center point
    height1 = round(height * 0.8)
    samples.append(face2DArray[height1][width])
    # store the hsv value at the bottom half center point
    height2 = round(height * 1.2)
    samples.append(face2DArray[height2][width])

    # convert to numpy array
    samples = np.asarray(samples)
    return samples


# define the skin tone range by selecting samples from the list
def define_skin_tone_range(skin_tone_samples):
    try:
        # degree of skew
        skew_degree = 2.2
        # calculate the first and last quarter sample selections and ensure int val
        lower = 3
        upper = 8
        # if we have reached the max samples
        # sort the list of hsv values
        sorted_samples = sorted(skin_tone_samples, key=lambda x: x[0])
        # take the value at position 50, which will be the lower skin tone range
        lower_skin = np.array(sorted_samples[lower-1:lower], dtype=np.uint8)
        # take the value at position 150, which will be the higher skin tone range
        upper_skin = np.array(sorted_samples[upper-1:upper], dtype=np.uint8)

        # define the list
        bounds = []

        # pre-set variables to define the range accepted, range further defined by extracted skin tone
        new_lower = np.array((0, lower_skin[0][1]/skew_degree, lower_skin[0][2]/skew_degree), dtype=np.uint8)
        bounds.append(new_lower)
        new_upper = np.array((upper_skin[0][0]/skew_degree, 255, 255), dtype=np.uint8)
        bounds.append(new_upper)

        # convert to numpy array
        bounds = np.asarray(bounds)

        # return the upper and lower HSV values
        return bounds
    except():
        return None