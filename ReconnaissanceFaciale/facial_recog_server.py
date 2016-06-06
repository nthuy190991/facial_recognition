# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 05:22:08 2016

@author: thnguyen
"""

import numpy as np
import os, sys
import cv2
import time
from read_xls import read_xls
#from edit_xls import edit_xls
import xlrd
from threading import Thread
from flask import Flask, request, render_template, Response
from processRequest import processRequest
import operator
from binascii import a2b_base64
import thread

"""
Replace French accents in texts
"""
def replace_accents(text):
    chars_origine = ['Ê','à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü']
    chars_replace  = ['\xC3','\xE0', '\xE1', '\xE2', '\xE3', '\xE4', '\xE5', '\xE6', '\xE7', '\xE8', '\xE9', '\xEA', '\xEB', '\xEC', '\xED', '\xEE', '\xEF', '\xF2', '\xF3', '\xF4', '\xF5', '\xF6', '\xF9', '\xFA', '\xFB', '\xFC']
    text2 = str_replace_chars(text, chars_origine, chars_replace)
    return text2

def replace_accents2(text):
    chars_origine = ['Ê','à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü']
    chars_replace  = ['E','a','a','a','a','a','a','ae','c','e','e','e','e','i','i','i','i','o','o','o','o','o','u','u','u','u']
    text2 = str_replace_chars(text, chars_origine, chars_replace)
    return text2
"""
Replace characters in a string
"""
def str_replace_chars(text, chars_origine, chars_replace):
    for i in range(len(chars_origine)):
        text2 = text.replace(chars_origine[i], chars_replace[i])
        text  = text2
    return text2

"""
==============================================================================
Face and Emotion API
==============================================================================
"""
def call_face_emotion_api(data):

    # Face API and Emotion API Variables
    _url_face   = 'https://api.projectoxford.ai/face/v1.0/detect'
    _key_face   = '5d99eec09a7e4b2a916eba7f75671600' # primary key
    _url_emo    = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
    _key_emo    = "b226d933ab854505b9b9877cf2f4ff7c" # primary key
    _maxNumRetries = 10

    global age, gender, emo
    #cv2.imwrite('testImageForAPI.jpg', img)

#    pathToFileInDisk = r'testImageForAPI.jpg'
#    with open( pathToFileInDisk, 'rb' ) as f:
#        data = f.read()

    """
    Face API
    """
    # Face detection parameters
    params = { 'returnFaceAttributes': 'age,gender,glasses',
           'returnFaceLandmarks': 'true'}
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/octet-stream'
    json = None

    faceResult = processRequest( _url_face, json, data, headers, params, _maxNumRetries )

    """
    Emotion API
    """
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_emo
    headers['Content-Type'] = 'application/octet-stream'

    emoResult = processRequest( _url_emo, json, data, headers, None, _maxNumRetries )

    """
    Results
    """
    print 'Found {} faces'.format(len(emoResult))
    nb_faces = len(faceResult)
    tb_age      = ['' for ind in range(nb_faces)]
    tb_gender   = ['' for ind in range(nb_faces)]
    tb_glasses  = ['' for ind in range(nb_faces)]
    tb_emo      = ['' for ind in range(len(emoResult))]

    if (len(faceResult)>0 and len(emoResult)>0):
        ind = 0
        for currFace in faceResult:
            #faceRectangle  = currFace['faceRectangle']
            faceAttributes  = currFace['faceAttributes']
            tb_age[ind]     = str(faceAttributes['age'])
            tb_gender[ind]  = faceAttributes['gender']
            tb_glasses[ind] = faceAttributes['glasses']
            ind += 1
        ind = 0
        for currFace in emoResult:
            tb_emo[ind] = max(currFace['scores'].iteritems(), key=operator.itemgetter(1))[0]
            ind += 1
        age     = tb_age[0]
        gender  = tb_gender[0]
        glasses = tb_glasses[0]
        emo     = tb_emo[0]
        return tb_age, tb_gender, tb_glasses, tb_emo

"""
Yield Face and Emotion API results
"""
def get_face_emotion_api_results():
    #global age, gender, emo
    resp = detect_face_attributes()
    if (resp==1):

        t0 = time.time()
        tb_age, tb_gender, tb_glasses, tb_emo = call_face_emotion_api(binary_data)

        # Translate emotion to french
        tb_emo_eng = ['happiness', 'sadness', 'surprise', 'anger', 'fear',
                  'contempt', 'disgust', 'neutral']
        tb_emo_correspond = ['êtes joyeux', 'êtes trist', 'êtes surprise',
                             'êtes en colère', 'avez peur', 'êtes mépris',
                             'êtes dégoût', 'êtes neutre']

        # Translate glasses to french
        tb_glasses_eng = ['NoGlasses', 'ReadingGlasses',
                      'sunglasses', 'swimmingGoggles']
        tb_glasses_correspond = ['ne portez pas de lunettes',
                                 'portez des lunettes',
                                 'portez des lunettes de soleil',
                                 'portez des lunettes de natation']

        for ind in range(len(tb_age)):
            glasses_str = tb_glasses_correspond[tb_glasses_eng.index(tb_glasses[ind])]
            emo_str     = tb_emo_correspond[tb_emo_eng.index(tb_emo[ind])]
            textToSpeak = "Bonjour " + ('Monsieur' if tb_gender[ind] =='male' else 'Madame') + \
                ", vous avez " + tb_age[ind].replace('.',',') + " ans, vous " + emo_str + \
                ", et vous " + glasses_str
            simple_message('Attributs faciales', textToSpeak)

        print 'Call Face and Emotion API: ' + str(time.time()-t0) + ' seconds'

#        cv2.waitKey(3000)
#        age    = ''
#        gender = ''
#        emo    = ''


"""
Read a .txt file
"""
def read_txt(filename):
    f = open(filename, "r")
    text = f.read()
    f.close()
    return text

"""
Read the first line of a .txt file
"""
def readline_txt(filename):
    f = open(filename, "r")
    text = f.readline()
    f.close()
    return text

"""
Write a text to .txt file
"""
def write_txt(filename, text):
    ManagedToOpenFile = False
    while not ManagedToOpenFile:
        try:
            f = open(filename, "w")
            ManagedToOpenFile = True
        except IOError:
            print("can't open file, retrying...")
    f.write(text)
    f.close()

"""
Ask a name or id as a string
"""
def ask_name(flag):
    global text, text2, text3, textFromHTML
    text  = ''
    text2 = ''
    text3 = "Donnez-moi votre identifiant, s'il vous plait !"

    if (flag):
        simple_message('', text3)

    while (textFromHTML==""):
        pass
    res = textFromHTML
    textFromHTML = ""

    return res

"""
Using Haar Cascade detector to detect faces from a grayscale image
"""
def detect_faces(faceCascade, gray):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (50, 50),
        flags        = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces


"""
Get all images in database alongside with their labels
"""
def get_images_and_labels(path, list_nom):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = [] # images will contains face images
    labels = [] # labels which are assigned to the image

    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Get the label of the image
        nom = os.path.split(image_path)[1].split(".")[0]
        if nom not in list_nom:
            list_nom.append(nom)

        nbr = list_nom.index(nom) + 1

        images.append(gray)
        labels.append(nbr)

    # return the images list and labels list
    return images, labels


"""
==============================================================================
Flask Initialization
==============================================================================
"""
def flask_init():
    global app, todo, stt, textFromHTML
    global clientId

    todo = ""
    stt  = ""
    textFromHTML = ""

    app  = Flask(__name__)

    @app.route('/')
    def render_hmtl():
        return render_template('index.html')

    @app.route('/start/<clientId>', methods=['POST'])
    def onStart(clientId):
        global thread_program
        global flag_thread_started
#        global HTML_refresh
#        HTML_refresh = True

        time.sleep(1)
        thread_program = Thread(target = run_program, name = 'thread_prog_'+clientId)
        thread_program.start()

        flag_thread_started = 1
        return "", 200

    @app.route('/StT/<data>', methods=['POST'])
    def runSpeechToText(data):
        clientId = data[0:4]
        text = data[5:]
        global stt
        stt = text
        #print 'StT', clientId, text
        return "", 200

    @app.route('/textFromHTML/<data>', methods=['POST'])
    def getTextFromHTML(data):
        global textFromHTML
        clientId = data[0:4]
        text = data[5:]
        textFromHTML = text
        #print 'textFromHTML', clientId, text
        return "", 200

    @app.route('/longpolling/<clientId>', methods=['POST'])
    def longPolling(clientId):
        time.sleep(0.5)
        global todo
        temp = todo
        #if temp!="": print temp
        todo = ""
        return temp, 200

    @app.route('/image', methods=['POST'])
    def getImage():
        global frameFromHTML, binary_data, clientId
        clientId        = request.get_json(force=True)["client_id"]
        image           = request.get_json(force=True)["img"]
        image           = image.split(",")[1]
        binary_data     = a2b_base64(image)
        data8uint       = np.fromstring(binary_data, np.uint8) # Convert bytestream to an unsigned int array
        frameFromHTML   = cv2.imdecode(data8uint, cv2.IMREAD_COLOR)
        return "",200


#    # TODO: Send back result to client
#    @app.route('/video_feed')
#    def video_feed():
#        _, jpeg = cv2.imencode('.jpg', frame)
#        return Response(b'--frame\r\n'
#           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes()
#           + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')


def flaskThread():
    app.run(host='0.0.0.0', port=5000)

"""
==============================================================================
Dialogue from Chrome
==============================================================================
"""
#def chrome_tts(text): # Text-to-Speech
#    global todo
#    todo = "TTS " + text # TODO: How to know when a text is finished speaking
#
#    # Calculate the time needed to wait, until the TTS is finished
#    text2 = str_replace_chars(text, [' ?',' !',' :',' ;'], ['?','!',':',';'])
#    nbOfWords  = len(text2.split())
#    rate = 1.1 # speech rate (which is set in hello.html)
#    timeNeeded = float(nbOfWords)/130/rate*60 # Average words-per-min = 130
#    time.sleep(timeNeeded)
#
#def chrome_stt(): # Speech-to-Text
#    global todo
#    global stt
#    stt  = ''
#    todo = 'STT'
#    t0   = time.time()
#    while (stt==''):
#        pass
#        if (time.time()-t0>=8): # Time outs after 8 secs
#            stt = '@' # Silence
#    resp = stt
#    return resp
#
#def chrome_yes_or_no(question):
#    chrome_tts(question) # Ask a question
#
#    global textFromHTML, todo
#    t0 = time.time()
#    while ((textFromHTML=="") and (time.time()-t0<5)):
#        pass
#    response = textFromHTML # Get answer from userInput during 5 seconds
#    textFromHTML = ""
#
#    if (response == ""):
#        response = chrome_stt() # Listen for an answer
#    else:
#        todo = "Confirm: " + response
#
#    if not(flag_quit):
#        if ('oui' in response):
#            result = 1
#        elif ('non' in response):
#            result = 0
#        elif (response == '@'):
#            result, response = chrome_yes_or_no("Je ne vous entends pas, veuillez répéter")
#            #chrome_tts("Je ne vous entends pas, veuillez répéter")
#            #response = chrome_stt()
#        else:
#            result, response = chrome_yes_or_no("Je ne vous comprends pas, veuillez répéter")
#            #chrome_tts("Je ne vous comprends pas, veuillez répéter")
#            #response = chrome_stt()
#        #chrome_tts("Vous avez répondu: " + response)
#    else:
#        result = -1
#        response = ''
#    return result, response


def chrome_tts(clientId, text): # Text-to-Speech
    #text = replace_accents2(text)
    global todo
    todo = "TTS/" + clientId + " " + text # TODO: How to know when a text is finished speaking

    # Calculate the time needed to wait, until the TTS is finished
    text2 = str_replace_chars(text, [' ?',' !',' :',' ;'], ['?','!',':',';'])
    nbOfWords  = len(text2.split())
    rate = 1.1 # speech rate (which is set in index.html)
    timeNeeded = float(nbOfWords)/130/rate*60 # Average words-per-min = 130
    time.sleep(timeNeeded)

def chrome_stt(clientId): # Speech-to-Text
    global todo
    global stt
    stt  = ''
    todo = 'STT'
    t0   = time.time()
    while (stt==''):
        pass
        if (time.time()-t0>=8): # Time outs after 8 secs
            stt = '@' # Silence
    resp = stt
    return resp

def chrome_yes_or_no(clientId, question):

    chrome_tts(clientId, question) # Ask a question

    global textFromHTML, todo
    t0 = time.time()
    while ((textFromHTML=="") and (time.time()-t0<3)):
        pass
    response = textFromHTML # Get answer from userInput during 3 seconds
    textFromHTML = ""

    if (response == ""):
        response = chrome_stt(clientId) # Listen for an answer
    else:
        todo = "Confirm: " + response

    if not(flag_quit):
        if ('oui' in response):
            result = 1
        elif ('non' in response):
            result = 0
        elif (response == '@'):
            result, response = chrome_yes_or_no(clientId, "Je ne vous entends pas, veuillez répéter")
            #chrome_tts("Je ne vous entends pas, veuillez répéter")
            #response = chrome_stt()
        else:
            result, response = chrome_yes_or_no(clientId, "Je ne vous comprends pas, veuillez répéter")
            #chrome_tts("Je ne vous comprends pas, veuillez répéter")
            #response = chrome_stt()
        #chrome_tts("Vous avez repondu: " + response)
    else:
        result = -1
        response = ''
    return result, response


"""
==============================================================================
Streaming Video: runs streaming video independently with other activities
==============================================================================
"""
def video_streaming():

    #global frame
    global image_save
    global key
    global tb_nb_times_recog
    global flag_quit

    time_origine = time.time()

    while True:
        frame = frameFromHTML # Get frame from HTML
        frame = cv2.flip(frame, 1) # Vertically flip frame
        key = cv2.waitKey(1)
        if (key == 27 or key2 == 27): # wait for ESC key to exit
            cv2.destroyAllWindows()
            flag_quit = 1
            break

        """
        Face Detection part
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to a grayscale image
        faces = detect_faces(faceCascade, gray) # Detect faces on grayscale image

        """
        Recognition part
        """
        for (x, y, w, h) in faces:
            if (len(faces)>1): # Consider only the biggest face appears in the video
                w_vect = faces.T[2,:]
                h_vect = faces.T[3,:]
                x0, y0, w0, h0 = faces[np.argmax(w_vect*h_vect)]
            elif (len(faces)==1): # If there is only one face
                x0, y0, w0, h0 = faces[0]

            if not flag_disable_detection:
                cv2.rectangle(frame, (x0, y0), (x0+w0, y0+h0), (25, 199, 247), 1) # Draw a rectangle around the biggest face
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (25, 199, 247), 1) # Draw a rectangle around the faces

            if (len(faces)>=1):
                image_save = gray[y0 : y0 + h0, x0 : x0 + w0]
                nbr_predicted, conf = recognizer.predict(image_save) # Predict function

                nom = list_nom[nbr_predicted-1] # Get resulting name

                if (conf < thres): # if recognizing distance is less than the predefined threshold -> FACE RECOGNIZED
                    if not flag_disable_detection:
                        txt = nom + ', distance: ' + str(conf)
                        message_xy(frame, txt, x0, y0-5, 'w', 1)

                    tb_nb_times_recog[nbr_predicted-1] = tb_nb_times_recog[nbr_predicted-1] + 1 # Increase nb of recognize times

                message_xy(frame, age, x0+w0, y0, 'b', 1)
                message_xy(frame, gender, x0+w0, y0+10, 'b', 1)
                message_xy(frame, emo, x0+w0, y0+20, 'b', 1)

        # End of For-loop

        # Texts to display on video
        count_time = time.time() - time_origine
        fps = count_fps()

        message(frame, "Time: " + str(count_time)[0:4], 0, 1, 'g', 2)
        message(frame, "FPS: "  + str(fps)[0:5],        0, 2, 'g', 1)
        message(frame, text,  0, 3, 'g', 1)
        message(frame, text2, 0, 4, 'g', 1)
        message(frame, text3, 0, 5, 'g', 1)

        # Frame display
        cv2.imshow('ClientId: ' + clientId + ' - Video streaming', frame)

    #cv2.destroyAllWindows()
    #quit_program()

"""
Put Texts on frame to display on streaming video at a predefined position (row,column)
"""
def message(frame, text, col, line, color, thickness):
    height, width = frame.shape[:2]
    if (col==0): x = 10
    if (line==1):
        y = 20
    elif (line==2):
        y = 40
    elif (line==3):
        y = height-50
    elif (line==4):
        y = height-30
    elif (line==5):
        y = height-10
    message_xy(frame, text, x, y, color, thickness)

"""
Put texts on frame to display on streaming video at position (x,y)
"""
def message_xy(frame, text, x, y, color, thickness):
    if color=='r':
        rgb = (0, 0, 255)
    elif color=='g':
        rgb = (0, 255, 0)
    elif color=='b':
        rgb = (255, 0, 0)
    elif color=='w':
        rgb = (255, 255, 255)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, rgb, thickness, lineType=cv2.CV_AA)


"""
Display Formation Panel for a recognized or username-known user
"""
def go_to_formation(xls_filename, name):

    global flag_disable_detection, flag_enable_recog, text, text2, text3
    flag_disable_detection = 1 # Disable the detection when entering Formation page
    flag_enable_recog = 0

    tb_formation = read_xls(xls_filename, 0) # Read Excel file which contains Formation info
    mail = reform_username(name) # Find email from name
    text = "Bonjour " + str(name)

    if (mail == '.'):
        text2 = "Votre information n'est pas disponible !"
        text3 = "Veuillez contacter contact@orange.com"
    else:
        mail_idx = tb_formation[0][:].index('Mail')

        # Get mail list
        mail_list = []
        for idx in range(0, len(tb_formation)):
            mail_list.append(tb_formation[idx][mail_idx])

        ind = mail_list.index(mail) # Find user in xls file based on his/her mail
        date = xlrd.xldate_as_tuple(tb_formation[ind][tb_formation[0][:].index('Date du jour')],0)
        text2 = "Bienvenue a la formation de "+str(tb_formation[ind][tb_formation[0][:].index('Prenom')])+" "+str(tb_formation[ind][tb_formation[0][:].index('Nom')] + ' !')
        text3 = "Vous avez un cours de " + str(tb_formation[ind][tb_formation[0][:].index('Formation')]) + ", dans la salle " + str(tb_formation[ind][tb_formation[0][:].index('Salle')]) + ", a partir du " + "{}/{}/{}".format(str(date[2]), str(date[1]),str(date[0]))

    simple_message('Page Formation', text2 + ' ' + text3)
    return text, text2, text3


"""
Return to recognition program after displaying Formation
"""
def return_to_recog():
    global flag_disable_detection, flag_enable_recog, flag_ask, flag_reidentify

    if not flag_quit:
        cv2.waitKey(5000)
        resp_quit_formation = quit_formation()
        if (resp_quit_formation == 0):
            time.sleep(15) # wait for more 15 seconds before quitting

        flag_disable_detection  = 0
        flag_enable_recog       = 1
        flag_ask = 1
        flag_reidentify = 0

"""
Find valid username
"""
def reform_username(name):

    if (name=='huy' or name=='huy_new'):
        firstname = 'thanhhuy'
        lastname = 'nguyen'
        email_suffix = '@orange.com'

    elif (name=='cleblain'):
        firstname = 'christian'
        lastname = 'leblainvaux'
        email_suffix = '@orange.com'

    elif (name=='catherine' or name=='lemarquis'):
        firstname = 'catherine'
        lastname = 'lemarquis'
        email_suffix = '@orange.com'

    elif (name=='ionel'):
        firstname = 'ionel'
        lastname = 'tothezan'
        email_suffix = '@orange.com'

    else:
        firstname = ''
        lastname = ''
        email_suffix = ''

    mail = firstname + '.' + lastname + email_suffix
    return mail


"""
==============================================================================
Taking photos
==============================================================================
"""
def take_photos(step_time, flag_show_photos):

    name = ask_name(1)
    #global flag_enable_recog
    global text, text2, text3

    image_to_paths = [imgPath+str(name)+"."+str(i)+suffix for i in range(nb_img_max)]

    if os.path.exists(imgPath+str(name)+".0"+suffix):
        print u"Les fichiers avec le nom " + str(name) + u" existent déjà"
        b = yes_or_no("Existence de fichiers", "Les fichiers avec le nom " + str(name) + " existent déjà, écraser ces fichiers ?", 3)
        if (b==1):
            for image_del_path in image_to_paths:
                os.remove(image_del_path)
        elif (b==0):
            name = ask_name(1)
            image_to_paths = [imgPath + str(name)+"."+str(i)+suffix for i in range(nb_img_max)]

    text = 'Prenant photos'
    text2 = 'Veuillez patienter... '

    simple_message('', text+', '+text2)

    nb_img = 0
    while (nb_img < nb_img_max):
        image_path = image_to_paths[nb_img]
        cv2.imwrite(image_path, image_save)
        print "Enregistrer photo " + image_path + ", nb de photos prises : " + str(nb_img+1)
        text3 = str(nb_img+1) + ' ont ete prises, reste a prendre : ' + str(nb_img_max-nb_img-1)
        nb_img += 1
        time.sleep(step_time)

    # Display photos that has just been taken
    if flag_show_photos:
        thread_show_photos = Thread(target = show_photos, args = (imgPath, name))
        thread_show_photos.start()

    time.sleep(0.25)

    # Allow to retake photos and validate after finish taking
    thread_retake_validate_photos = Thread(target = retake_validate_photos, args = (step_time, flag_show_photos, imgPath, name))
    thread_retake_validate_photos.start()


"""
Retaking and validating photos
"""
def retake_validate_photos(step_time, flag_show_photos, imgPath, name):

    global flag_enable_recog, flag_ask, flag_wrong_recog
    global text, text2, text3

    # Ask users if they want to change photo(s) or validate them
    b = validate_photo()
    image_to_paths = [root_path+imgPath+str(name)+"."+str(j)+suffix for j in range(nb_img_max)]

    while (b==0):
        text3 = "Veuillez repondre sur Terminal"
        simple_message('', "Veuillez répondre quelles photos que vous voulez changer ?")

        global textFromHTML
        while textFromHTML=="":
            pass
        nb = textFromHTML
        textFromHTML = ""

        if ('-' in nb):
            nb2 = ''
            for i in range(int(nb[0]), int(nb[2])+1):
                nb2 = nb2+str(i)
            nb=nb2

        nb = str_replace_chars(nb, [',',';','.',' '], ['','','',''])

        if (nb!=""):
            str_nb= ""
            for j in range(0, len(nb)):
                if (j==len(nb)-1):
                    str_nb = str_nb + "'" + nb[j] + "'"
                else:
                    str_nb = str_nb + "'" + nb[j] + "', "

            simple_message('Reprise de photos', 'Vous souhaitez changer les photos: ' + str_nb + ' ?')

            text  = 'Prenant photos'
            text2 = 'Veuillez patienter... '
            text3 = ''

        for j in range(0, len(nb)):
            text3 = str(j) + ' ont ete prises, reste a prendre : ' + str(len(nb)-j)
            time.sleep(step_time)
            print "Reprendre photo ", nb[j]
            image_path = image_to_paths[int(nb[j])-1]
            os.remove(image_path) # Remove old image
            cv2.imwrite(image_path, image_save)
            print "Enregistrer photo " + image_path + ", nb de photos prises : " + nb[j]

        a = yes_or_no('Nouvelles photos', 'Reprise de photos finie, souhaitez-vous réviser vos photos ?', 4)
        if (a==1):
            thread_show_photos2 = Thread(target = show_photos, args = (imgPath, name))
            thread_show_photos2.start()

        b = validate_photo()
        text  = ''
        text2 = ''
        text3 = ''
        if (b==1):
            break
    # End of While(b==0)

    # Update recognizer after taking and validating photos
    images, labels = get_images_and_labels(imgPath, list_nom)
    recognizer.update(images, np.array(labels))
    print u"Recognizer a été mis a jour..."

    flag_enable_recog = 1  # Re-enable recognition
    #flag_wrong_recog = 0 # Reset wrong recognition flag
    flag_ask = 1 # Reset asking

"""
Display photos that have just been taken, close them if after 5 seconds or press any key
"""
def show_photos(imgPath, name):
    x = 100; y = 600

    image_to_paths = [root_path + imgPath + str(name) + "." + str(j) + suffix for j in range(nb_img_max)]

    ind = 1
    for img_path in image_to_paths:
        #print img_path
        img = cv2.imread(img_path)
        cv2.imshow('Photo '+str(ind), img)
        height, width = img.shape[:2]
        cv2.moveWindow('Photo '+str(ind), x, y)
        x=x+width
        ind += 1

    cv2.waitKey(7000) # wait a key for 7 seconds
    for ind in range(nb_img_max):
        cv2.destroyWindow('Photo '+str(ind+1))

"""
==============================================================================
Re-identification: when a user is not recognized or not correctly recognized
==============================================================================
"""
def re_identification(nb_time_max):

    simple_message('Autre positionnement', 'Pouvez-vous rapprocher vers la camera, ou bouger votre tête ?')

    global flag_enable_recog, flag_ask, flag_take_photo, flag_wrong_recog, flag_reidentify#, flag_thread_photo
    tb_old_name    = np.chararray(shape=(nb_time_max+1), itemsize=10) # Old recognition results, which are wrong
    tb_old_name[:] = ''
    tb_old_name[0] = name0

    nb_time = 0
    flag_enable_recog = 1
    flag_reidentify   = 1
    time.sleep(wait_time) # wait until after the first re-identification is done
    flag_ask = 0
    a = 0
    while (nb_time < nb_time_max):
        #time.sleep(2)
        name1 = nom # New result
        #print tb_old_name
        # TODO: if unknown person --> Count instead of retrying (done, but should be verified)

        if np.all(tb_old_name != name1) and flag_recog: # if new result is different to old results
            print 'Essaie ' + str(nb_time+1) + ': reconnu comme ' + str(name1)
            if (a==0):
                a = 1 # Small trick to not to ask twice (not start two Speech Recognizer) at the same time
                resp = validate_recognition()
            if (resp == 1):
                a = 0
                result = 1
                name = name1
                break
            else:
                result = 0
                a = 0
                nb_time += 1
                tb_old_name[nb_time] = name1
                time.sleep(wait_time)

        elif (not flag_recog):
            print 'Essaie ' + str(nb_time+1) + ': personne inconnue'
            a = 0
            result = 0
            nb_time += 1
            time.sleep(wait_time)

    if (result==1): # User confirms that the recognition is correct now
        flag_enable_recog = 0
        flag_reidentify   = 0
        flag_wrong_recog  = 0

        get_face_emotion_api_results()

        text, text2, text3 = go_to_formation(xls_filename, name)

        return_to_recog() # Return to recognition program immediately or 20 seconds before returning

    else: # Two time failed to recognized
        flag_enable_recog = 0 # Disable recognition when two tries have failed
        flag_reidentify   = 0
        simple_message('Problème méconnaissable', 'Désolé je vous reconnaît pas, veuillez me donner votre identifiant')

        name = ask_name(0)
        if os.path.exists(imgPath+str(name)+".0"+suffix): # Assume that user's face-database exists if the photo 0.png exists
            simple_message('Reprise de photos', 'Bonjour '+ str(name)+', je vous conseille de changer vos photos')
            flag_show_photos = 1
            step_time = 1

            thread_show_photos3 = Thread(target = show_photos, args = (imgPath, name))
            thread_show_photos3.start()

            time.sleep(0.25)
            thread_retake_validate_photos2 = Thread(target = retake_validate_photos, args = (step_time, flag_show_photos, imgPath, name))
            thread_retake_validate_photos2.start()
        else:
            simple_message('Erreur', "Malheureusement, les photos correspondant au nom "+ str(name) +" n'existent pas. Je vous conseille de reprendre vos photos")
            flag_take_photo  = 1  # Enable photo taking

"""
==============================================================================
Main program body with decision and redirection
==============================================================================
"""
def run_program():

    # Global variables
    global flag_enable_recog, flag_ask, flag_recog, flag_wrong_recog, flag_reidentify, flag_take_photo, flag_quit
    global text, text2, text3
    global tb_nb_times_recog, nom, name0
    global key2
    global thread_video

    # Autorisation to begin Streaming Video
    optin0 = allow_streaming_video()

    if (optin0==1):
        key = 0
        flag_quit = 0

        # Thread of streaming video
        thread_video = Thread(target = video_streaming, name = 'thread_video_'+clientId)
        thread_video.start()

        # Table of number of times that a face is correctly recognized
        tb_nb_times_recog = np.empty(len(list_nom))
        tb_nb_times_recog.fill(0) # initialize with all zeros

        time.sleep(0.25) # Time needed to start the thread of streaming video

        start_time   = time.time() # For recognition timer (will reset after each 3 secs)
        time_origine = time.time() # For display (unchanged)

        """
        Permanent loop
        """
        while True:

            # Break While-loop and quit program as soon as the Esc key is pressed
            if (key==27):
                time.sleep(0.5)
                thread_video.join()
                print 'Join streaming video thread'
                break

            """
            Decision part
            """
            if not (flag_quit): #TODO: new
                key2 = cv2.waitKey(1)
                if (key2 == 27):# or HTML_refresh:
                    break
                elapsed_time = time.time() - start_time
                if ((elapsed_time > wait_time) and flag_enable_recog): # Identify after each 3 seconds
                    if ((max(tb_nb_times_recog) >= nb_max_times/2)): # If the number of times recognized is big enough
                        flag_recog = 1 # Known Person
                        flag_ask = 0
                        nom = list_nom[np.argmax(tb_nb_times_recog)] # Get name of recognizing face

                        text  = 'Reconnu : ' + nom
                        if (not flag_reidentify):
                            text2 = "Appuyez [Y] si c'est bien vous"
                            text3 = "Appuyez [N] si ce n'est pas vous"

                            res_verify_recog = verify_recog()
                            if (res_verify_recog==1):
                                key = ord('y')
                            elif (res_verify_recog==0):
                                key = ord('n')

                    else: # If the number of times recognized anyone from database is too low
                        flag_recog = 0 # Unknown Person
                        #nom = '' # XXX: new: à vérifier
                        text  = 'Personne inconnue'
                        text2 = ''
                        text3 = ''
                        if (not flag_reidentify):
                            flag_ask = 1
                            simple_message('', "Désolé, je ne vous reconnaît pas")

                    tb_nb_times_recog.fill(0) # reinitialize with all zeros
                    start_time = time.time()  # reset timer

                """
                Redirecting user based on recognition result and user's status (already took photos or not) in database
                """
                count_time = time.time() - time_origine
                if (count_time <= wait_time):
                    text3 = 'Initialisation (pret dans ' + str(wait_time-count_time)[0:4] + ' secondes)...'
                    if (flag_quit):
                        time.sleep(0.5)
                        thread_video.join()
                        print "Let's get the hell out of here"
                        break
                else:
                    """
                    Start Redirecting after the first 3 seconds
                    """
                    if (flag_quit):
                        time.sleep(0.5)
                        thread_video.join()
                        print "Let's get the hell out of here"
                        break
                    if (flag_recog):
                        if (key==ord('y') or key==ord('Y')): # User chooses Y to go to Formation page
                            flag_wrong_recog  = 0
                            get_face_emotion_api_results()
                            text, text2, text3 = go_to_formation(xls_filename, nom)

                            key = 0
                            return_to_recog() # Return to recognition program, after displaying Formation

                        if (key==ord('n') or key==ord('N')): # User confirms that the recognition result is wrong by choosing N
                            flag_wrong_recog = 1
                            flag_ask = 1
                            key = 0
        #                    thres = thres - 10 # Reduce threshold
        #                    print u'Réduire threshold à ' + str(thres)

                    if ((flag_recog and flag_wrong_recog) or (not flag_recog)): # Not recognized or not correctly recognized
                        if (flag_ask):# and (not flag_quit)):
                            resp_deja_photos = deja_photos() # Ask user if he has already had a database of face photos

                            if (resp_deja_photos==-1):
                                flag_ask = 0

                            elif (resp_deja_photos==1): # User has a database of photos
                                flag_enable_recog = 0 # Disable recognition in order not to recognize while re-identifying
                                flag_ask = 0

                                name0 = nom     # Save the recognition result, which is wrong, in order to compare later
                                nb_time_max = 2 # Number of times to retry recognize

                                thread_reidentification = Thread(target = re_identification, args = (nb_time_max,))
                                thread_reidentification.start()

                            elif (resp_deja_photos==0): # User doesnt have a database of photos

                                flag_enable_recog = 0 # Disable recognition in order not to recognize while taking photos
                                resp_allow_take_photos = allow_take_photos()

                                if (resp_allow_take_photos==1): # User allows to take photos
                                    flag_take_photo = 1  # Enable photo taking
                                    #flag_enable_recog = 0 # Stop recognition while taking photos

                                else: # User doesnt want to take photos
                                    flag_take_photo = 0
                                    res = allow_go_to_formation_by_id()
                                    if (res==1): # User agrees to go to Formation in providing his id manually
                                        name = ask_name(1)
                                        text, text2, text3 = go_to_formation(xls_filename, name)
                                        #flag_enable_recog = 0

                                        # Return to recognition program (if user wishs to, otherwise, wait 20 seconds before returning anyway)
                                        return_to_recog()

                                    else: # Quit if user refuses to provide manually his id (after all other functionalities)
                                        break
                                        #quit_program()

                                resp_allow_take_photos = 0
                            resp_deja_photos = 0
                        flag_ask = 0

                    if (flag_take_photo):# and (not flag_quit)):

                        step_time  = 1 # Interval of time (in second) between two times of taking photo

                        thread_take_photo = Thread(target = take_photos, args = (step_time, 1))
                        thread_take_photo.start()

                        tb_nb_times_recog = np.empty(len(list_nom)+1) # Extend the list with one more value for the new face
                        tb_nb_times_recog.fill(0) # reinitialize the table with all zeros
                        #flag_recog = 1 # the recognizer is supposed to know the new person who has just been taken photos
                        flag_take_photo = 0

                    """
                    Call Face API and Emotion API, and display
                    """
                    if (key==ord('i') or key==ord('I')):
                        call_face_emotion_api(binary_data)
                        key=0
        """
        End of While-loop
        """
    """
    Exit the program
    """
    quit_program()
#    flag_quit = 0
#    chrome_tts("Merci de votre utilisation. Au revoir, à bientôt")
#    flag_quit = 1
#    global todo
#    todo = ""
#    sys.exit()


"""
Quit program
"""
def quit_program():
     global todo
     todo = "quit"
     time.sleep(0.25)

     global flag_quit
     flag_quit = 0 # Turn it on to execute just the yes_no question and bye-bye
     quit_opt = yes_or_no('Exit', 'Voulez-vous vraiment quitter ?', 3) # This wont executed if quit by Esc key
     #cv2.destroyAllWindows()

     chrome_tts(clientId, "Merci de votre utilisation. Au revoir, à bientôt")
     flag_quit = 1
     todo = ""

     if (quit_opt == 0):
          print 'Relance programme...'
          import facial_recog_server
          reload(facial_recog_server) # Reload program
     elif (quit_opt == 1):
         print u'Session a terminée, fermer programme...'
         #sys.exit() # Supplementary to quit program more correctly



"""
Definition of used yes-no questions in program
"""
def detect_face_attributes():
    resp = yes_or_no("", "Voulez-vous apercevoir vos attributs faciales ?", 3)
    return resp

def verify_recog():
    resp = yes_or_no("", "Bonjour " + nom + ", est-ce bien vous ?", 3)
    return resp

def allow_streaming_video():
    resp = yes_or_no('Video Streaming', 'Bonjour ! Autorisez-vous le lancement de la video streaming ?', 4)
    return resp

def deja_photos():
    resp = yes_or_no('Base de photos', 'Avez-vous déjà pris des photos ?', 3)
    return resp

def allow_take_photos():
    resp = yes_or_no('Prise de photos', "Êtes-vous d'accord pour vous faire prendre en photos ?", 3)
    return resp

def validate_photo():
    resp = yes_or_no('Validation de photos', 'Voulez-vous valider ces photos ?', 4)
    return resp

def allow_go_to_formation_by_id():
    resp = yes_or_no('Accès Formation', "Voulez-vous accéder votre page Formation par votre identifiant ?", 3)
    return resp

def quit_formation():
    resp = yes_or_no('Quitter cette page', 'Voulez-vous quitter la page Formation ?', 4)
    return resp

def validate_recognition():
    resp = yes_or_no("Re-identification", "Bonjour " + nom + ", est-ce bien vous cette fois-ci ?", 4)
    return resp

"""
Yes/No question as an asking/answering by dialogue
"""
def yes_or_no(title, message, type_message_box):
    if (not flag_quit): # Put in If-condition to allow interrupt when Esc is pressed
        #print 'not good'
        resp, ouinon = chrome_yes_or_no(clientId, message)
        return resp
    else:
        return -1

"""
Simple message as a notification speech
"""
def simple_message(title, message):
    if (not flag_quit): # Put in If-condition to allow interrupt when Esc is pressed
        chrome_tts(clientId, message)

"""
Calculating Frame-per-second parameter
"""
def count_fps():
    video = cv2.VideoCapture(-1)
    fps    = video.get(cv2.cv.CV_CAP_PROP_FPS);
    return fps



"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
# Parameters
root_path    = ""
cascPath     = "haarcascade_frontalface_default.xml" # path to Haar-cascade training xml file
imgPath      = "face_database/" # path to database of faces
suffix       = '.png' # image file extention
thres        = 80     # Distance threshold for recognition
wait_time    = 2      # Time needed to wait for recognition
nb_max_times = 10     # Maximum number of times of good recognition counted in 3 seconds (manually determined, and depends on camera)
nb_img_max   = 5      # Number of photos needs to be taken for each user
xls_filename = 'formation' # Excel file contains Formation information
nom = ''

# Messages to appear on streaming video (at line 3, 4, 5)
text    = ''
text2   = ''
text3   = ''
age     = ''
gender  = ''
emo     = ''

# Flags used in program
flag_recog        = 0 # Recognition flag (flag=1 if recognize someone, flag=0 otherwise)
flag_take_photo   = 0 # Flag if unknown user chooses to take photos
flag_wrong_recog  = 0 # Flag if a person is recognized but not correctly, and feedbacks
flag_enable_recog = 1 # Flag of enabling or not the recognition
flag_disable_detection = 0 # Flag of disabling displaying the detection during some other task (Formation, Taking photos)
flag_quit         = 0
flag_ask          = 0 # Flag if it is necessary to ask 'etes vous dans ma base ou pas ?'
flag_reidentify   = 0

# Haar cascade detector used for face detection
faceCascade = cv2.CascadeClassifier(root_path + cascPath)

# For face recognition we use the Local Binary Pattern Histogram (LBPH) Face Recognizer
recognizer  = cv2.createLBPHFaceRecognizer()
list_nom    = []

# Call the get_images_and_labels function and get the face images and the corresponding labels
print u"Obtenu Images et Labels à partir de database..."
images, labels = get_images_and_labels(root_path + imgPath, list_nom)

# Perform the training
recognizer.train(images, np.array(labels))
print u"Apprentissage a été fini...\n"

# Initialisation global variables
#frame  = 0
image_save = 0
key   = 0 # Quit key inside video streaming thread
key2  = 0 # Quit key from run program

clientId = "xxxx"

todo = ''
flask_init()

#HTML_refresh = False

#app.run(host='0.0.0.0', port=5000, threaded=True)

thread.start_new_thread(flaskThread,())

flag_thread_started  = 0

while True:
    if flag_thread_started:
        if (thread_program.is_alive() == False):
            print 'join run_program thread\n'
            thread_program.join()
            flag_thread_started = 0

# END OF PROGRAM
