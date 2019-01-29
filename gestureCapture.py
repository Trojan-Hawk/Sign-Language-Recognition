# Imports
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

# define the default range of skin color
lower_skin = np.array([0, 0, 0], dtype=np.uint8)
upper_skin = np.array([255, 255, 255], dtype=np.uint8)

# Difference of 220 between start and end gives an adequate region of interest
regionStartX = 250
regionStartY = 130
regionEndX = 470
regionEndY = 350

# counter
global count
count = 0
# boolean control
skinToneOK = False
# skin tone list
skinToneSamples = []

# external XML file which defines the cascade face detection classifier
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


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
    # take the hsv value at that point
    print(face2DArray[height][width])
    skinToneSamples.append(face2DArray[height][width])


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
    except():
        print("Cannot find face!")
        return False


max_samples = 10


# define the skin tone range by selecting samples from the list
def define_skin_tone_range():
    # degree of skew
    skew_degree = 2.2
    # link to global variables
    global lower_skin, upper_skin
    # calculate the first and last quarter sample selections and ensure int val
    lower = int(round((max_samples / 100) * 25))
    upper = int(round((max_samples / 100) * 75))
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
    new_upper = np.array((upper_skin[0][0], 255, 255), dtype=np.uint8)
    upper_skin = new_upper

while(1):
    # try to display data to the user
    try:
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
            if mask is not None:
                print("Mask is NONE!!!!")
                # lower_skin = np.array([0, 48, 80], dtype= np.uint8)
                # upper_skin = np.array([20, 255, 255], dtype= np.uint8)
                # skin colour range
                mask = cv2.inRange(hsv, lower_skin, upper_skin)


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