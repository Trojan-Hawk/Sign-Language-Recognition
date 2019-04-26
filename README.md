# Sign-Language-Recognition
### Video Demo
The video demo can be found at the following URL:
https://youtu.be/HOtiEZAmvAI

### Python Script Descriptions
The Python scripts contained in this directory are briefly explained below:
##### GestureApplication.py
This is the script containing the full client side project, with implemented graphical user interface.
##### GestureCapture.py
This is the script that was initially created to capture gesture frames using the OpenCV lirary.
##### NeuralNetwork.py
This is the script that was created to develop the artificial neural network aspect of this project.
##### GestureRecognition.py
This is the script that was initially created to combine the GestureCapture.py script and the NeuralNetwork.py script.
##### CreateTrainingDirectory.py
This is the script that was created to read in all images from the dataset directory, pickle the datasets and then export them as pickled files.  
##### SaveNormalizedDatasets.py
This is the script that was created to fix the normalization issue encountered on the server.
##### facial_recognition.py
This is the script that implements the haar cascade classifier algorithm.
##### NNServer.py
This script is the multi-threaded server which allows multi-user connections. The purpose of this file is to host the neural network so it is not coupled to the client application.
### File Descriptions
The files contained in this directory are briefly explained below:
##### FYP_Dissertation.pdf
This file is the Minor Dissertation associated with this repository.
##### Saved_Weights directory
This directory contains all the pre-trained weight files, which can be reused to mirror error rate.
##### Server Directory
This is the directory where the server script is located.
##### haarcascade_frontalface_default.xml
This is the feature detection classifier file used in the facial_recognition Python script.
##### default.jpg
This is the default image used when the application encounters an error.
### Datasets

The source of the normalized training and testing datasets can be found here: https://drive.google.com/file/d/1SDyD9tlbMj9e21WXCYuWpncXpDSuuIBT/view?usp=sharing
