import os, sys
import time
import numpy as np
import cv2
from faceRecognizerFuncs import getLabelsInfo

"""
Using Haar Cascade detector to detect faces from a grayscale image
"""
def detect_faces(faceCascade, gray):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 3,
        minSize      = (100, 100),
        flags        = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces


"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
try:
    filename = sys.argv[1]
    if (filename == '-h') or (filename == '--help'):
        print 'mdl_recognition_opencv.py <recognizer_xml>'
        sys.exit()

except IndexError:
    print "Set parameters as default"
    filename = "faceRecognizer.xml"

# Parameters
root_path   = ""
cascPath    = "haarcascade_frontalface_default.xml" # path to Haar-cascade training xml file
imgPath     = "face_database/" # path to database of faces
suffix      = '.png' # image file extention

# Haar cascade detector used for face detection
faceCascade = cv2.CascadeClassifier(root_path + cascPath)

# For face recognition we use the Local Binary Pattern Histogram (LBPH) Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()

# Load recognizer from xml
if "xml" not in filename:
	filename = filename + ".xml"
recognizer.load(filename)

# Get Labels info
labelsInfoDict = getLabelsInfo(filename)
print labelsInfoDict

# Video streaming
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    
    if (ret == True):
        frame = cv2.flip(frame, 1) # Vertically flip frame

        key = cv2.waitKey(1)
        if (key == 27): # wait for ESC key to exit
            break

        """
        Face Detection
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to a grayscale image
        faces = detect_faces(faceCascade, gray) # Detect faces on grayscale image

        if (len(faces)>1): # Consider only the biggest face appears in the video
            w_vect = faces.T[2,:]
            h_vect = faces.T[3,:]
            x0, y0, w0, h0 = faces[np.argmax(w_vect*h_vect)]
        elif (len(faces)==1): # If there is only one face
            x0, y0, w0, h0 = faces[0]

        if len(faces)>=1:
            cv2.rectangle(frame, (x0, y0), (x0+w0, y0+h0), (255, 0, 0), 1) # Draw a rectangle around the biggest face
            image_save = gray[y0 : y0 + h0, x0 : x0 + w0]
            
            [p_label, p_confidence] = recognizer.predict(image_save)

            name_predict = labelsInfoDict[p_label]
            text = "Predicted: %s (confidence=%.2f)" % (name_predict, p_confidence)
            cv2.putText(frame, text, (x0+10, y0-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1, lineType=cv2.CV_AA)

            print "Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence)

        # Frame display
    cv2.imshow('Video streaming', frame)

video_capture.release() # Release video capture
cv2.destroyWindow('Video streaming')