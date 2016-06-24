import os, sys
import cv2
import time
import face_api
from threading import Thread


def create_group_add_person(groupId, groupName):

    # Create PersonGroup
    result = face_api.createPersonGroup(groupId, groupName, "")
    flag_reuse_person_group = False
    if (result!=''):
        result = eval(result)
        
        if (result["error"]["code"] == "PersonGroupExists"):
            
            del_person_group = raw_input('Delete this PersonGroup? (y/n) ')
    
            if (del_person_group=='y') or (del_person_group=='1'):
                print 'PersonGroup exists, deleting...'
        
                res_del = face_api.deletePersonGroup(groupId)
                print ('Deleting PersonGroup succeeded' if res_del=='' else 'Deleting PersonGroup failed')
                
                result = face_api.createPersonGroup(groupId, groupName, "")
                print ('Re-create PersonGroup succeeded' if res_del=='' else 'Re-create PersonGroup failed')
                
            elif (del_person_group=='n') or (del_person_group=='0'):
                # Get PersonGroup training status
                res      = face_api.getPersonGroupTrainingStatus(groupId)
                res      = res.replace('null','None')
                res_dict = eval(res)
                training_status = res_dict['status']
                if (training_status=='succeeded'):
                    flag_reuse_person_group = True
        elif (result["error"]["code"] == "RateLimitExceeded"):
            print 'RateLimitExceeded, please retry after 30 seconds'
            sys.exit()
    
    if not flag_reuse_person_group:
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

def faceRecongize(faceDetectResult):
  
    global text, x0, y0, w0, h0
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
            name_predict         = recognizedPerson['name']

            text = "Recognized: %s (confidence=%.2f)" % (name_predict, recognizedConfidence)
            print text
                
"""
==============================================================================
Recognition on video
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
    wait_time  = 4 # Identify after each 4 seconds
    image_path = 'video.jpg'
    
    start_time   = time.time() # For recognition timer (will reset after each 3 secs)
    time_origine = time.time() # For counting using time
    
    thread_video = Thread(target = streaming_video, name = 'thread_video')
    thread_video.start()
    
    while (time.time() - time_origine < 20) and (key != 27):
        elapsed_time = time.time() - start_time
        if (elapsed_time > wait_time): # Identify after each 4 seconds
        
            faceDetectResult = face_api.faceDetect(None, image_path, None)
            faceRecongize(faceDetectResult)
                
            start_time = time.time()  # reset timer
   

"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
try:
    imgPath = sys.argv[1]
    if (imgPath == '-h') or (imgPath == '--help'):
        print 'mdl_recognition_msoxford.py <path_to_database> <path_to_image_test/video>'
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

create_group_add_person(groupId, groupName)

result = train_person_group(groupId)


if (not flag_video):
    faceDetectResult    = face_api.faceDetect(None, imgTest, None)
    faceRecongize(faceDetectResult)
    
else:
    text, x0, y0, w0, h0 = '', 0, 0, 0, 0
    key = 0
    recognize_from_video()

    
    