import os, sys
import time
import numpy as np
import cv2
from threading import Thread

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
Display photos that have just been taken, close them if after 5 seconds or press any key
"""
def show_photos(imgPath, name):
    x = 100; y = 600

    image_to_paths = [root_path + imgPath + str(name) + "." + str(j) + suffix for j in range(nbImages)]

    ind = 1
    for img_path in image_to_paths:
        img = cv2.imread(img_path)
        cv2.imshow('Photo '+str(ind), img)
        height, width = img.shape[:2]
        cv2.moveWindow('Photo '+str(ind), x, y)
        x   += width
        ind += 1

    cv2.waitKey(5000) # wait a key for 7 seconds
    for ind in range(nbImages):
        cv2.destroyWindow('Photo '+str(ind+1))


"""
==============================================================================
Saving photos
==============================================================================
"""
def save_photo(name, idxImg):
    
    image_to_path = imgPath+str(name)+"."+str(idxImg)+suffix

    # Erase photo if existed
    if os.path.exists(image_to_path):
        os.remove(image_to_path)
       
    cv2.imwrite(image_to_path, image_save)
    print "Saving photo " + image_to_path + ", nb of photos taken: " + str(idxImg+1)


"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
name = ''
try:
    name = sys.argv[1]
    if (name == '-h') or (name == '--help'):
        print 'mdl_take_photos_from_photos.py <username> <path_to_images>'
        print 'If there are more than 1 image, please put them in the same folder'
        sys.exit()
        
    imagePath = sys.argv[2]
        
except IndexError:
    print "Set parameters as default"
    if (name==''):
        name = 'new'
    imagePath = os.getcwd()
        
root_path   = ""
cascPath    = "haarcascade_frontalface_default.xml" # path to Haar-cascade training xml file
imgPath     = "face_database/" # path to database of faces
suffix      = '.png' # image file extention

# Haar cascade detector used for face detection
faceCascade = cv2.CascadeClassifier(root_path + cascPath)

flag_show_photos = True # flag to show photos after being taken

image_paths = [os.path.join(imagePath, f) for f in os.listdir(imagePath) if (f.endswith('.png') or f.endswith('.jpg'))]
idxImg      = 0
nbImages    = len(image_paths)

for image_path in image_paths:
    frame = cv2.imread(image_path)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to a grayscale image
    faces = detect_faces(faceCascade, gray) # Detect faces on grayscale image

    if (len(faces)==1): # If there is only one face
        x0, y0, w0, h0 = faces[0]
        image_save = gray[y0 : y0 + h0, x0 : x0 + w0]
        save_photo(name, idxImg)
        idxImg += 1

    elif (len(faces)>1): # Consider only the biggest face appears in the video
        print 'Error: Found more than one face in image ' + image_path
        sys.exit()

# Display photos that have just been taken
if flag_show_photos:
    thread_show_photos = Thread(target = show_photos, args = (imgPath, name), name = 'thread_show_photos')
    thread_show_photos.start()
