# Imports
import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
     
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

        # Output of the visual data input (Camera)
        print(cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

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

        defectCount += 1
        
        # font style and size variables
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 2

        # print corresponding gestures which are in their ranges
        if defectCount == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Place hand(s) in the box', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'Thumbs up!', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
                   
                else:
                    cv2.putText(frame, '1', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
                    
        elif defectCount == 2:
            cv2.putText(frame, '2', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
            
        elif defectCount == 3:
         
            if arearatio < 27:
                    cv2.putText(frame, '3', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                    cv2.putText(frame, 'ok', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
                    
        elif defectCount == 4:
            cv2.putText(frame, '4', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
            
        elif defectCount == 5:
            cv2.putText(frame, '5', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
            
        elif defectCount == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)
            
        else:
            cv2.putText(frame, 'reposition', (10, 50), font, fontScale, (0, 0, 255), 3, cv2.LINE_AA)

        print("Above imgshow")

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