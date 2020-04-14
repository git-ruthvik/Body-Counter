import numpy as np
import cv2
import datetime


file=open('bodynumbers.txt','w')
# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture(1)

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    #Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Draw rectangles
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    #Dispay window
    cv2.imshow('camera', frame)

    #Count and write into file
    count = str(len(bodies))
    num = str('Time={}, Count={}'.format(datetime.datetime.now(),count))
    file.write(num)
    file.write('\n')

    #Press Enter Key to exit stream
    if cv2.waitKey(1) == 13:
        break

# Release video stream
cap.release()
cv2.destroyAllWindows()
