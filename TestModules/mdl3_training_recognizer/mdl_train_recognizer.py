import os, sys
import numpy as np
import cv2
from faceRecognizerFuncs import setLabelsInfo

"""
Get all images in database alongside with their labels
"""
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    list_nom = []
    images 	 = [] # images will contains face images
    labels 	 = [] # labels which are assigned to the image

    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Get the label of the image
        nom = os.path.split(image_path)[1].split(".")[0]
        if nom not in list_nom:
            list_nom.append(nom)

        nbr = list_nom.index(nom)

        images.append(gray)
        labels.append(nbr)

    # return the images list and labels list
    return images, labels, list_nom

"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
try:
    imgPath = sys.argv[1]
    if (imgPath == '-h') or (imgPath == '--help'):
        print 'mdl_train_recognizer.py <face_database> <output_xml>'
        sys.exit()
    filename = sys.argv[2]
        
except IndexError:
    print "Set parameters as default"
    imgPath = "face_database/" # path to database of faces
    filename = "faceRecognizer.xml"

# Parameters
root_path = ""
suffix    = '.png' # image file extention

# For face recognition we use the Local Binary Pattern Histogram (LBPH) Face Recognizer
recognizer  = cv2.createLBPHFaceRecognizer()

# Get the face images and the corresponding labels
images, labels, list_nom = get_images_and_labels(root_path + imgPath)
print "Get images and labels from database..."

# Train recognizer
recognizer.train(images, np.array(labels))
print "FaceRecognizer training finished..."

# Save recognizer to xml
if "xml" not in filename:
	filename = filename + ".xml"
recognizer.save(filename)
print 'Saved recognizer to ' + filename

# Add supplementary info to labels
labelsInfo = dict([])
for nom in list_nom:
    labelsInfo.update({list_nom.index(nom): nom})

setLabelsInfo(filename, labelsInfo)