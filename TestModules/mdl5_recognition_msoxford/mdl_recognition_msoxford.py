import os, sys
import cv2
import time
import face_api
from threading import Thread


def create_group_add_person(groupId, groupName):

    # Delete PersonGroup that existed
    face_api.deletePersonGroup(groupId)

    # Create PersonGroup
    result = face_api.createPersonGroup(groupId, groupName, "")
    print result

    # if result["error"]["code"] == "PersonGroupExists":
    #     #TODO: change to get personGroup info than delete it
    #     #face_api.deletePersonGroup(groupId)
    #     print 'a'

    # Create person and add person image
    image_paths = [os.path.join(imgPath, f) for f in os.listdir(imgPath)]
    nbr = 0
    
    for image_path in image_paths:
        nom = os.path.split(image_path)[1].split(".")[0]
        if nom not in list_nom:
            # Create a Person in PersonGroup
            personName = nom
            personId   = face_api.createPerson(groupId, personName, "")

            list_nom.append(nom)
            list_personId.append(personId)
            nbr += 1
        else:
            personId = list_personId[nbr-1]

        # Add image
        face_api.addPersonFace(groupId, personId, None, image_path, None)
        print "Add image...", nom, '\t', image_path
        time.sleep(0.25)


def train_person_group(groupId):

    # Train PersonGroup
    face_api.trainPersonGroup(groupId)

    # Get training status
    res      = face_api.getPersonGroupTrainingStatus(groupId)
    res      = res.replace('null','None')
    res_dict = eval(res)
    training_status = res_dict['status']
    print training_status

    while (training_status=='running'):
        time.sleep(0.25)
        res = face_api.getPersonGroupTrainingStatus(groupId)
        res = res.replace('null','None')
        res_dict = eval(res)
        training_status = res_dict['status']
        print training_status

    return training_status


"""
==============================================================================
Recognize from video
==============================================================================
"""
def streaming_video():
    global key
    image_path = 'video.jpg'
    
    # Video streaming
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        
        if (ret == True):
            frame = cv2.flip(frame, 1) # Vertically flip frame
            cv2.imwrite(image_path, frame)
            
            key = cv2.waitKey(1)
            if (key == 27): # wait for ESC key to exit
                break
            
            #cv2.putText(frame, 'something', (200, 200), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1, lineType=cv2.CV_AA)
            
            if (w0!=0):                   
                cv2.rectangle(frame, (x0, y0), (x0+w0, y0+h0), (255, 0, 0), 1) # Draw a rectangle around the biggest face
                cv2.putText(frame, text, (x0+10, y0-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 1, lineType=cv2.CV_AA)
            
            # Frame display
        cv2.imshow('Video streaming', frame)
    """
    End of While-loop
    """
    
    video_capture.release() # Release video capture
    cv2.destroyWindow('Video streaming')
    
    
def recognize_from_video():
    global text, x0, y0, w0, h0
    key = 0
    wait_time  = 4
    image_path = 'video.jpg'
    
    start_time   = time.time() # For recognition timer (will reset after each 3 secs)
    time_origine = time.time() # For counting using time
    
    thread_video = Thread(target = streaming_video, name = 'thread_video')
    thread_video.start()
    
    while (time.time() - time_origine < 10) and (key != 27):
        
        elapsed_time = time.time() - start_time
        if (elapsed_time > wait_time): # Identify after each 4 seconds
            faceDetectResult    = face_api.faceDetect(None, image_path, None)

            if (len(faceDetectResult)>=1):
                faceRectangle  = faceDetectResult[0]['faceRectangle']    
                x0, y0, w0, h0 = faceRectangle['left'], faceRectangle['top'], faceRectangle['width'], faceRectangle['height']
                new_faceId     = faceDetectResult[0]['faceId']
                resultIdentify = face_api.faceIdentify(groupId, [new_faceId], maxNbOfCandidates)

                if (len(resultIdentify[0]['candidates'])>=1):

                    recognizedPersonId   = resultIdentify[0]['candidates'][0]['personId']
                    recognizedConfidence = resultIdentify[0]['candidates'][0]['confidence']
                    recognizedPerson     = face_api.getPerson(groupId, recognizedPersonId)
                    recognizedPerson     = recognizedPerson.replace('null','None')
                    recognizedPerson     = eval(recognizedPerson)
                    name_predict   = recognizedPerson['name']

                    text = "Recognized: %s (confidence=%.2f)" % (name_predict, recognizedConfidence)
                    
                    print text
                    print x0, y0, w0, h0
                    
                start_time = time.time()  # reset timer

    """
    End of While-loop
    """


    

"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
try:
    imgPath = sys.argv[1]
    if (imgPath == '-h') or (imgPath == '--help'):
        print 'mdl_recognition_msoxford.py <path_to_database> <path_to_image_test>'
        sys.exit()       
    imgTest = sys.argv[2]
    if (imgTest=='video'):
        flag_video = True
    else: 
        flag_video = False
        
except IndexError:
    print "Set parameters as default"
    imgPath = "face_database_for_oxford/" # path to database of faces
    imgTest = 'test.png'
    flag_video = False

# Parameters     
maxNbOfCandidates = 1 # Maximum number of candidates for the identification

# Training Phase
groupId     = "group_all"
groupName   = "employeurs"

list_nom = []
list_personId = []
nbr = 0

#create_group_add_person(groupId, groupName)
#
#result = train_person_group(groupId)


if (not flag_video):
    faceDetectResult    = face_api.faceDetect(None, imgTest, None)
    
    if (len(faceDetectResult)>=1):
        new_faceId      = faceDetectResult[0]['faceId']
        resultIdentify  = face_api.faceIdentify(groupId, [new_faceId], maxNbOfCandidates)
    
        if (len(resultIdentify[0]['candidates'])>=1): # If the number of times recognized is big enough
            recognizedPersonId  = resultIdentify[0]['candidates'][0]['personId']
            recognizedConfidence = resultIdentify[0]['candidates'][0]['confidence']
            recognizedPerson    = face_api.getPerson(groupId, recognizedPersonId)
            recognizedPerson    = recognizedPerson.replace('null','None')
            recognizedPerson    = eval(recognizedPerson)
            name_predict   = recognizedPerson['name']
    
            print "Recognized: %s (confidence=%.2f)" % (name_predict, recognizedConfidence)
else:
    text, x0, y0, w0, h0 = '', 0, 0, 0, 0
    recognize_from_video()
#    thread_video = Thread(target = recognize_from_video, name = 'thread_video')
#    thread_video.start()
    
    
    