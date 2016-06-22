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
def save_photos(name, step_time, flag_show_photos):
    while not video_start:
        pass
    
    image_to_paths = [imgPath+str(name)+"."+str(i)+suffix for i in range(nbImages)]

    # Erase photos if existed
    if os.path.exists(imgPath+str(name)+".0"+suffix):
        for image_del_path in image_to_paths:
            os.remove(image_del_path)
       
    idxImg = 0
    while (idxImg < nbImages):
        image_path = image_to_paths[idxImg]
        cv2.imwrite(image_path, image_save)
        print "Saving photo " + image_path + ", nb of photos taken: " + str(idxImg+1)
        idxImg += 1
        time.sleep(step_time)
        
    global flag_finish
    flag_finish = True
    
    # Display photos that have just been taken
    if flag_show_photos:
        thread_show_photos = Thread(target = show_photos, args = (imgPath, name), name = 'thread_show_photos')
        thread_show_photos.start()


"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""

try:
    name = sys.argv[1]
    if (name == '-h') or (name == '--help'):
        print 'mdl_take_photos_from_video.py <filename> <nb_of_photos> <taking_time>'
        print 'For example: mdl_take_photo.py person1 5 5'
        sys.exit()
        
    nbImages = int(sys.argv[2])
    temps    = int(sys.argv[3])
        
except IndexError:
    print "Set parameters as default"
    name = 'new'
    nbImages = 5
    temps = 5
    
    
root_path   = ""
cascPath    = "haarcascade_frontalface_default.xml" # path to Haar-cascade training xml file
imgPath     = "face_database/" # path to database of faces
suffix      = '.png' # image file extention

# Haar cascade detector used for face detection
faceCascade = cv2.CascadeClassifier(root_path + cascPath)

video_start = False
step_time   = float(temps)/float(nbImages) # Time between taking two photos
flag_show_photos = True # flag to show photos after being taken

flag_finish = False
thread_save_photos = Thread(target = save_photos, args = (name, step_time, flag_show_photos), name = 'thread_save_photos')
thread_save_photos.start()


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    
    if (ret == True):
        frame = cv2.flip(frame, 1) # Vertically flip frame

        key = cv2.waitKey(1)
        if (key == 27 or flag_finish): # wait for ESC key to exit
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
            
            video_start = True

        # Frame display
    cv2.imshow('Video streaming', frame)

video_capture.release() # Release video capture
cv2.destroyWindow('Video streaming')

