from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
# facial recognition script import
from facial_recognition import facial_recognition as face_rec
from facial_recognition import define_skin_tone_range as hsv_bounds
import socket
import blosc
import pickle

# canvas size
HEIGHT = 1000
WIDTH = 1800
# slider range
LOW = 0
HIGH = 255
# slider length
LENGTH = 400
# font style and size variables
font = cv2.FONT_HERSHEY_SIMPLEX

# Region of interest bounds, 220 between start and end gives an adequate region of interest
regionStartX = 250
regionStartY = 130
regionEndX = 470
regionEndY = 350

# openCV video capture
cap = cv2.VideoCapture(0)

# global hsv bounds
global lower_hsv, upper_hsv
# global hsv list
global skinToneSamples
skin_tone_samples = []
# boolean controls
preforming_facial_recognition = True
settings_visible = False

# consecutive prediction global variables
global last_prediction
last_prediction = ''
global counter
counter = 0
global MAX_CONSECUTIVE
MAX_CONSECUTIVE = 10

# initialize the socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# the host address and port to be used
HOST = '18.221.79.199'      # The server's public IP address
PORT = 7777                 # The port used by the server
try:
    # open the connection
    s.connect((HOST, PORT))
except:
    print("Cannot Connect to server!")
    pass


def capture_current_frame():
    # get a handle on the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # while frame has no data
    while frame is None:
        # get a handle on the camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    return frame


def define_HSV_bounds():
    global lower_hsv, upper_hsv
    counter = 4
    while counter is not 0:
        frame = capture_current_frame()
        # retrieve some hsv samples from the current frame
        samples = face_rec(frame)

        if samples is not None:
            # iterate over the array and store the values on a list
            for i in samples:
                skin_tone_samples.append(i)
            counter = counter-1

    # extract an upper and lower hsv value from the list
    bounds = hsv_bounds(skin_tone_samples)
    # update the lower and upper bounds
    lower_hsv = bounds[0]
    upper_hsv = bounds[1]

    # update the GUI sliders
    scaleHLower.set(lower_hsv[0]);scaleSLower.set(lower_hsv[1]);scaleVLower.set(lower_hsv[2]);
    scaleHUpper.set(upper_hsv[0]);scaleSUpper.set(upper_hsv[1]);scaleVUpper.set(upper_hsv[2]);


def show_camera():
    try:
        frame = capture_current_frame()

        # draw the region of interest
        cv2.rectangle(frame, (regionStartY, regionStartX), (regionEndY, regionEndX), (0, 255, 0), 0)
        # extract region of interest
        roi = frame[regionStartX:regionEndX, regionStartY:regionEndY]
        # show the extracted image
        show_extracted(roi)

        # convert the array to an image
        image = Image.fromarray(frame)
        # cast to a PhotoImage object
        photo = ImageTk.PhotoImage(image=image)
        # update the camera panel
        cameraPanel.imgtk = photo
        cameraPanel.configure(image=photo)
        cameraPanel.after(10, show_camera)
    except ValueError:
        print("ValueError.")
    except():
        print("GUI Issue: cameraPanel")
        pass


def show_extracted(roi):
    try:
        kernel = np.ones((3, 3), np.uint8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # skin colour range
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=2)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        # convert the array to an image
        img = Image.fromarray(mask)
        # cast to a PhotoImage object
        photo = ImageTk.PhotoImage(image=img)
        # update the camera panel
        resultPanel.imgtk = photo
        resultPanel.configure(image=photo)

        # get the server prediction and update GUI
        get_server_prediction(roi)
    except():
        print("GUI Issue: resultPanel")


def get_server_prediction(roi):
    # send the region of interest
    s.sendall(roi)
    # get the response(encoded, pickled and compressed)
    data = s.recv(300000)
    # decompress
    data = blosc.decompress(data)
    # un-pickle
    data = pickle.loads(data)
    # decode
    data = data.decode('utf-8')
    # update the GUI
    print(data)


def update_H_value_lower(value):
    if int(value) < upper_hsv[0]:       lower_hsv[0] = value
    else:                               lower_hsv[0] = upper_hsv[0]; hVarLower.set(upper_hsv[0])
def update_S_value_lower(value):
    if int(value) < upper_hsv[1]:       lower_hsv[1] = value
    else:                               lower_hsv[1] = upper_hsv[1]; sVarLower.set(upper_hsv[1])
def update_V_value_lower(value):
    if int(value) < upper_hsv[2]:       lower_hsv[2] = value
    else:                               lower_hsv[2] = upper_hsv[2]; vVarLower.set(upper_hsv[2])
def update_H_value_upper(value):
    if int(value) > lower_hsv[0]:       upper_hsv[0] = value
    else:                               upper_hsv[0] = lower_hsv[0]; hVarUpper.set(lower_hsv[0])
def update_S_value_upper(value):
    if int(value) > lower_hsv[1]:       upper_hsv[1] = value
    else:                               upper_hsv[1] = lower_hsv[1]; sVarUpper.set(lower_hsv[1])
def update_V_value_upper(value):
    if int(value) > lower_hsv[2]:       upper_hsv[2] = value
    else:                               upper_hsv[2] = lower_hsv[2]; vVarUpper.set(lower_hsv[2])


def is_consecutive(prediction):
    global counter
    if prediction == last_prediction:
        counter = counter+1
    if counter >= MAX_CONSECUTIVE:
        counter = 0
        update_predicted_gui(prediction)


def update_predicted_gui(prediction):
    # allow the gui to be edited
    outputText.config(state=NORMAL)
    # inserting the text
    outputText.insert(INSERT, prediction)
    # prevent the user from editing the text
    outputText.config(state=DISABLED)
    # packing the contents
    outputLabel.pack()


def toggle_settings():
    global settings_visible
    settings_visible = not settings_visible


# root window for the application GUI
root = Tk()

# default image used to display output
img = 'default.jpg'
default_img = ImageTk.PhotoImage(Image.open(img))

# setting the size of the canvas
canvas = Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

# camera frame
cameraFrame = Frame(root, bg='#80c1ff')
cameraFrame.place(relwidth=0.4, relheight=0.6, relx=0.05, rely=0.05)
# label
cameraLabel = Label(cameraFrame, text='Camera Frame', bg='black', fg='white')
cameraLabel.pack()
# image panel
cameraPanel = Label(cameraFrame, image=default_img, bg='black')
cameraPanel.pack(side="bottom", fill="both", expand="yes")

# result frame
resultFrame = Frame(root, bg='#80c1ff')
resultFrame.place(relwidth=0.3, relheight=0.4, relx=0.60, rely=0.05)
# label
resultLabel = Label(resultFrame, text='Result Frame', bg='black', fg='white')
resultLabel.pack()
# image panel
resultPanel = Label(resultFrame, image=default_img, bg='black')
resultPanel.pack(side="bottom", fill="both", expand="yes")

# Lower HSV sliders frame
slidersFrameLower = Frame(root, bg='#80c1ff')
slidersFrameLower.place(relwidth=0.3, relheight=0.2, relx=0.60, rely=0.50)
# label
slidersLabelLower = Label(slidersFrameLower, text='HSV Lower Bounds', bg='black', fg='white')
slidersLabelLower.pack()
# H, S and V sliders (upper & lower)
hVarLower = DoubleVar()
scaleHLower = Scale(slidersFrameLower, length=LENGTH, variable=hVarLower, orient=HORIZONTAL, from_=LOW, to=HIGH, command=update_H_value_lower)
scaleHLower.pack(anchor=CENTER, pady=6)
sVarLower = DoubleVar()
scaleSLower = Scale(slidersFrameLower, length=LENGTH, variable=sVarLower, orient=HORIZONTAL, from_=LOW, to=HIGH, command=update_S_value_lower)
scaleSLower.pack(anchor=CENTER, pady=6)
vVarLower = DoubleVar()
scaleVLower = Scale(slidersFrameLower, length=LENGTH, variable=vVarLower, orient=HORIZONTAL, from_=LOW, to=HIGH-155, command=update_V_value_lower)
scaleVLower.pack(anchor=CENTER, pady=6)

# Upper HSV sliders frame
slidersFrameUpper = Frame(root, bg='#80c1ff')
slidersFrameUpper.place(relwidth=0.3, relheight=0.2, relx=0.60, rely=0.75)
# label
slidersLabelUpper = Label(slidersFrameUpper, text='HSV Upper Bounds', bg='black', fg='white')
slidersLabelUpper.pack()

hVarUpper = DoubleVar()
scaleHUpper = Scale(slidersFrameUpper, length=LENGTH, variable=hVarUpper, orient=HORIZONTAL, from_=LOW, to=HIGH, command=update_H_value_upper)
scaleHUpper.pack(anchor=CENTER, pady=6)
sVarUpper = DoubleVar()
scaleSUpper = Scale(slidersFrameUpper, length=LENGTH, variable=sVarUpper, orient=HORIZONTAL, from_=LOW, to=HIGH, command=update_S_value_upper)
scaleSUpper.pack(anchor=CENTER, pady=6)
vVarUpper = DoubleVar()
scaleVUpper = Scale(slidersFrameUpper, length=LENGTH, variable=vVarUpper, orient=HORIZONTAL, from_=LOW, to=HIGH-155, command=update_V_value_upper)
scaleVUpper.pack(anchor=CENTER, pady=6)

# Text output frame
# result frame
outputFrame = Frame(root, bg='#80c1ff')
outputFrame.place(relwidth=0.4, relheight=0.2, relx=0.05, rely=0.75)
# label
outputLabel = Label(outputFrame, text='Predictions', bg='black', fg='white')
# Text area
outputText = Text('', font=(font, 13), padx=25, pady=25, width=65, height=12, wrap=WORD)
# prevent the user from editing the text
outputText.config(state=DISABLED)
# place text
outputText.place(relwidth=0.36, relheight=0.14, relx=0.069, rely=0.78)
outputLabel.pack()

# button which will call the facial recognition function
button = Button(canvas, justify=CENTER, text="Facial Recognition", command=define_HSV_bounds)
# pack the button onto the canvas
button.pack()

# call the specified method after 10 milliseconds
if preforming_facial_recognition:
    root.after(10, define_HSV_bounds)
    # update the control boolean
    preforming_facial_recognition = False
if not preforming_facial_recognition:
    root.after(10, show_camera)

# start the GUI main loop
root.mainloop()

# On exit, remove all windows created
cv2.destroyAllWindows()
# and release the camera input stream
cap.release()
