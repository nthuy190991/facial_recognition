# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 05:22:08 2016

@author: thnguyen
"""

import numpy as np
import os
import time
from read_xls import read_xls
#from edit_xls import edit_xls
import xlrd
from threading import Thread
from flask import Flask, request, render_template#, Response
import operator
from binascii import a2b_base64
from watson_developer_cloud import NaturalLanguageClassifierV1
import face_api
import emotion_api
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
Replace French accents in texts
"""
def replace_accents(text):
    chars_origine = ['Ê','à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü']
    chars_replace = ['\xC3','\xE0', '\xE1', '\xE2', '\xE3', '\xE4', '\xE5', '\xE6', '\xE7', '\xE8', '\xE9', '\xEA', '\xEB', '\xEC', '\xED', '\xEE', '\xEF', '\xF2', '\xF3', '\xF4', '\xF5', '\xF6', '\xF9', '\xFA', '\xFB', '\xFC']
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
def retrieve_face_emotion_att(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    data = global_var['binary_data']
        
    # Face API
    faceResult = face_api.faceDetect(None, None, data)
    
    # Emotion API
    emoResult = emotion_api.recognizeEmotion(None, None, data)

    # Results
    print 'Found {} '.format(len(faceResult)) + ('faces' if len(faceResult)!=1 else 'face')
    nb_faces = len(faceResult)
    tb_face_rect = [{} for ind in range(nb_faces)]
    tb_age       = ['' for ind in range(nb_faces)]
    tb_gender    = ['' for ind in range(nb_faces)]
    tb_glasses   = ['' for ind in range(nb_faces)]
    tb_emo       = ['' for ind in range(len(emoResult))]

    if (len(faceResult)>0 and len(emoResult)>0):
        ind = 0
        for currFace in faceResult:
            faceRectangle       = currFace['faceRectangle']
            faceAttributes      = currFace['faceAttributes']
            
            tb_face_rect[ind]   = faceRectangle
            tb_age[ind]         = str(faceAttributes['age'])
            tb_gender[ind]      = faceAttributes['gender']
            tb_glasses[ind]     = faceAttributes['glasses']
            ind += 1
            
        ind = 0
        for currFace in emoResult:
            tb_emo[ind] = max(currFace['scores'].iteritems(), key=operator.itemgetter(1))[0]
            ind += 1
            
        faceWidth  = np.zeros(shape=(nb_faces))
        faceHeight = np.zeros(shape=(nb_faces))
        for ind in range(nb_faces):
            faceWidth[ind]  = tb_face_rect[ind]['width']
            faceHeight[ind] = tb_face_rect[ind]['height']
        ind_max = np.argmax(faceWidth*faceHeight.T)
        
        global_var['age']     = tb_age[ind_max]
        global_var['gender']  = tb_gender[ind_max]
        global_var['emo']     = tb_emo[ind_max]
        
#        global_var['age']     = tb_age[0] # TODO: replace the first face by the biggest face (Done)
#        global_var['gender']  = tb_gender[0]
#        global_var['emo']     = tb_emo[0]

        return tb_age, tb_gender, tb_glasses, tb_emo
    else:
        return 'N/A','N/A','N/A','N/A'

"""
Yield Face and Emotion API results
"""
def get_face_emotion_api_results(clientId):

    resp = detect_face_attributes(clientId)
    if (resp==1):

        print 'Calling APIs to retrieve facial and emotional attributes, please wait'
        tb_age, tb_gender, tb_glasses, tb_emo = retrieve_face_emotion_att(clientId)

        if ([tb_age, tb_gender, tb_glasses, tb_emo] != ['N/A','N/A','N/A','N/A']):
            # Translate emotion to french
            tb_emo_eng = ['happiness', 'sadness', 'surprise', 'anger', 'fear',
                          'contempt', 'disgust', 'neutral']
            tb_emo_correspond = ['joyeux', 'trist', 'surprise',
                                 'en colère', "d'avoir peur", ' mépris',
                                 'dégoût', 'neutre']
    
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
                    ", vous avez " + tb_age[ind].replace('.',',') + " ans, votre état d'émotion est " + emo_str + \
                    ", et vous " + glasses_str
                simple_message(clientId, 'Attributs faciales', textToSpeak)
            # print 'Call Face and Emotion API: ' + str(time.time()-t0) + ' seconds'
        else:
            print 'Found no faces'
            simple_message(clientId, 'Attributs faciales', u'Désolé, aucun visage trouvé')

"""
Ask a name or id as a string
"""
def ask_name(clientId, flag):
    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    global_var['text']  = ''
    global_var['text2'] = ''
    global_var['text3'] = "Donnez-moi votre identifiant, s'il vous plait !"

    if (flag):
        simple_message(clientId, '', global_var['text3'])

    while (global_var['textFromHTML']==""):
        pass
    res = global_var['textFromHTML']
    global_var['textFromHTML'] = ""

    return res


"""
==============================================================================
Dialogue from Chrome
==============================================================================
"""

def chrome_tts(clientId, text): # Text-to-Speech
    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    global_var['todo'] = "TTS " + str(clientId) + " " + text
    global_var['tts']  = text

    # Calculate the time needed to wait, until the TTS is finished
    text2 = str_replace_chars(text, [' ?',' !',' :',' ;'], ['?','!',':',';'])
    nbOfWords  = len(text2.split())
    rate = 1.2 # speech rate (which is set in index.html)
    timeNeeded = float(nbOfWords)/130/rate*60 # Average words-per-min = 130
    time.sleep(timeNeeded)

def chrome_stt(clientId): # Speech-to-Text
    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    global_var['todo'] = 'STT'
    global_var['stt']  = ''

    t0 = time.time()
    while (global_var['stt'] == ''):
        pass
        if (time.time()-t0>=7): # Time outs after 8 secs
            global_var['stt'] = '@' # Silence
    resp = global_var['stt']
    return resp

def chrome_yes_or_no(clientId, question):

    chrome_tts(clientId, question) # Ask a question

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    t0 = time.time()
    while ((global_var['textFromHTML']=="") and (time.time()-t0<5)):
        pass
    response = global_var['textFromHTML'] # Get answer from userInput during 3 seconds
    global_var['textFromHTML'] = ""

    if (response == ""):
        response = chrome_stt(clientId) # Listen for an answer

    if (response == '@'):
        result, response = chrome_yes_or_no(clientId, u"Je ne vous entends pas, veuillez répéter")

    # Apply Natural Language Classifier
    classes = natural_language_classifier.classify('2374f9x68-nlc-1265', response)
    responseYesOrNo = classes["top_class"]

    if not(global_var['flag_quit']):
        if (responseYesOrNo=='oui'):
            result = 1
        elif (responseYesOrNo=='non'):
            result = 0
        elif (responseYesOrNo=='not_relevant'):
            result, response = chrome_yes_or_no(clientId, u"Votre réponse n'est pas pertinente, veuillez ré-répondre")
    else:
        result   = -1
        responseYesOrNo = ''
    return result, response


"""
==============================================================================
Display Formation Panel for a recognized or username-known user
==============================================================================
"""
def go_to_formation(clientId, xls_filename, name):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    global_var['flag_disable_detection'] = 1 # Disable the detection when entering Formation page
    global_var['flag_enable_recog']      = 0

    tb_formation = read_xls(xls_filename, 0) # Read Excel file which contains Formation info
    mail = reform_username(name) # Find email from name
    global_var['text'] = "Bonjour " + str(name)

    if (mail == '.'):
        global_var['text2'] = "Votre information n'est pas disponible !"
        global_var['text3'] = "Veuillez contacter contact@orange.com"
    else:
        mail_idx = tb_formation[0][:].index('Mail')
        # Get mail list
        mail_list = []
        for idx in range(0, len(tb_formation)):
            mail_list.append(tb_formation[idx][mail_idx])

        ind = mail_list.index(mail) # Find user in xls file based on his/her mail
        date = xlrd.xldate_as_tuple(tb_formation[ind][tb_formation[0][:].index('Date du jour')],0)
        global_var['text2'] = "Bienvenue a la formation de "+str(tb_formation[ind][tb_formation[0][:].index('Prenom')])+" "+str(tb_formation[ind][tb_formation[0][:].index('Nom')] + ' !')
        global_var['text3'] = "Vous avez un cours de " + str(tb_formation[ind][tb_formation[0][:].index('Formation')]) + ", dans la salle " + str(tb_formation[ind][tb_formation[0][:].index('Salle')]) + ", a partir du " + "{}/{}/{}".format(str(date[2]), str(date[1]),str(date[0]))

    simple_message(clientId, 'Page Formation', global_var['text2'] + ' ' + global_var['text3'])
    return global_var['text'], global_var['text2'], global_var['text3']


"""
Return to recognition program after displaying Formation
"""
def return_to_recog(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    if not global_var['flag_quit']:
        resp_quit_formation = quit_formation(clientId)
        if (resp_quit_formation == 0):
            time.sleep(5) # wait for more 5 seconds before quitting

        global_var['flag_disable_detection']  = 0
        global_var['flag_enable_recog']       = 1
        global_var['flag_ask']                = 1
        global_var['flag_reidentify']         = 0

"""
Find valid username
"""
def reform_username(name):

    if (name=='huy' or name=='GGQN0871'):
        firstname    = 'thanhhuy'
        lastname     = 'nguyen'
        email_suffix = '@orange.com'

    elif (name=='cleblain'):
        firstname    = 'christian'
        lastname     = 'leblainvaux'
        email_suffix = '@orange.com'

    elif (name=='catherine' or name=='lemarquis'):
        firstname    = 'catherine'
        lastname     = 'lemarquis'
        email_suffix = '@orange.com'

    elif (name=='ionel'):
        firstname    = 'ionel'
        lastname     = 'tothezan'
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
def take_photos(clientId, step_time, flag_show_photos):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    name = ask_name(clientId, 1)

    personId = face_api.createPerson(groupId, name, "")
    list_personId.append(personId)

    image_to_paths = [imgPath+str(name)+"."+str(i)+suffix for i in range(nb_img_max)]

    if os.path.exists(imgPath+str(name)+".0"+suffix):
        print u"Les fichiers avec le nom " + str(name) + u" existent déjà"
        b = yes_or_no(clientId,"Existence de fichiers", u"Les fichiers avec le nom " + str(name) + u" existent déjà, écraser ces fichiers ?", 3)
        if (b==1):
            for image_del_path in image_to_paths:
                os.remove(image_del_path)
        elif (b==0):
            name = ask_name(clientId, 1)
            image_to_paths = [imgPath + str(name)+"."+str(i)+suffix for i in range(nb_img_max)]

    global_var['text']  = 'Prenant photos'
    global_var['text2'] = 'Veuillez patienter... '

    simple_message(clientId, '', global_var['text'] + ', ' + global_var['text2'])

    nb_img = 0
    while (nb_img < nb_img_max):
        image_path = image_to_paths[nb_img]
        with open(image_path, 'wb') as f:
            f.write(global_var['binary_data'])
            f.close()

        print "Enregistrer photo " + image_path + ", nb de photos prises : " + str(nb_img+1)
        global_var['text3'] = str(nb_img+1) + ' ont ete prises, reste a prendre : ' + str(nb_img_max-nb_img-1)
        nb_img += 1
        time.sleep(step_time)

    # Display photos that has just been taken
    if flag_show_photos:
        thread_show_photos = Thread(target = show_photos, args = (clientId, imgPath, name), name = 'thread_show_photos_'+clientId)
        thread_show_photos.start()

    time.sleep(0.5)

    # Allow to retake photos and validate after finish taking
    thread_retake_validate_photos = Thread(target = retake_validate_photos, args = (clientId, personId, step_time, flag_show_photos, imgPath, name), name = 'thread_retake_validate_photos_'+clientId)
    thread_retake_validate_photos.start()


"""
Retaking and validating photos
"""
def retake_validate_photos(clientId, personId, step_time, flag_show_photos, imgPath, name):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    # Ask users if they want to change photo(s) or validate them
    b = validate_photo(clientId)
    image_to_paths = [root_path+imgPath+str(name)+"."+str(j)+suffix for j in range(nb_img_max)]

    while (b==0):
        global_var['text3'] = "Veuillez repondre"
        simple_message(clientId, '', u"Veuillez répondre quelles photos que vous voulez changer ?")

        while (global_var['textFromHTML'] == ""):
            pass
        nb = global_var['textFromHTML']
        global_var['textFromHTML'] = ""

        if ('-' in nb):
            nb2 = ''
            for i in range(int(nb[0]), int(nb[2])+1):
                nb2 = nb2 + str(i)
            nb = nb2

        nb = str_replace_chars(nb, [',',';','.',' '], ['','','',''])

        if (nb!=""):
            str_nb = ""
            for j in range(0, len(nb)):
                if (j==len(nb)-1):
                    str_nb = str_nb + "'" + nb[j] + "'"
                else:
                    str_nb = str_nb + "'" + nb[j] + "', "

            simple_message(clientId, 'Reprise de photos', 'Vous souhaitez changer les photos: ' + str_nb + ' ?')

            global_var['text']  = 'Prenant photos'
            global_var['text2'] = 'Veuillez patienter... '
            global_var['text3'] = ''

        for j in range(0, len(nb)):
            global_var['text3'] = str(j) + ' ont ete prises, reste a prendre : ' + str(len(nb)-j)
            time.sleep(step_time)
            print "Reprendre photo ", nb[j]
            #TODO: add a facedetect here to cut the face from image
            image_path = image_to_paths[int(nb[j])-1]
            os.remove(image_path) # Remove old image
            with open(image_path, 'wb') as f:
                f.write(global_var['binary_data'])
                f.close()

            print "Enregistrer photo " + image_path + ", nb de photos prises : " + nb[j]

        a = yes_or_no(clientId, 'Nouvelles photos', u'Reprise de photos finie, souhaitez-vous réviser vos photos ?', 4)
        if (a==1):
            thread_show_photos2 = Thread(target = show_photos, args = (clientId, imgPath, name), name = 'thread_show_photos2_'+clientId)
            thread_show_photos2.start()

        b = validate_photo(clientId)
        global_var['text']  = ''
        global_var['text2'] = ''
        global_var['text3'] = ''
        if (b==1):
            break
    # End of While(b==0)

    print "Adding faces to person group..."
    image_to_paths = [root_path+imgPath+str(name)+"."+str(j)+suffix for j in range(nb_img_max)]
    for image_path in image_to_paths:
        face_api.addPersonFace(groupId, personId, None, image_path, None)

    # Retrain Person Group
    resultTrainPersonGroup = face_api.trainPersonGroup(groupId)
    print "Re-train Person Group: ", resultTrainPersonGroup
    
    global_var['flag_enable_recog'] = 1  # Re-enable recognition
    global_var['flag_ask'] = 1 # Reset asking


"""
Display photos that have just been taken, close them if after 5 seconds or press any key
"""
def show_photos(clientId, imgPath, name):

    image_to_paths = [root_path + imgPath + str(name) + "." + str(j) + suffix for j in range(nb_img_max)]

    for img_path in image_to_paths:
        print 'display', img_path
        plt.figure()
        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()

    time.sleep(5) # wait 5 secs
    for ind in range(nb_img_max):
        plt.close("all")
        
        
"""
==============================================================================
Re-identification: when a user is not recognized or not correctly recognized
==============================================================================
"""
def re_identification(clientId, nb_time_max, name0):

    simple_message(clientId, 'Autre positionnement', u'Veuillez rapprocher vers la camera, ou bouger votre tête')

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    tb_old_name    = np.chararray(shape=(nb_time_max+1), itemsize=10) # Old recognition results, which are wrong
    tb_old_name[:] = ''
    tb_old_name[0] = name0

    nb_time = 0
    global_var['flag_enable_recog'] = 1
    global_var['flag_reidentify']   = 1
    global_var['flag_ask'] = 0

    time.sleep(wait_time) # wait until after the first re-identification is done

    a = 0
    while (nb_time < nb_time_max):
        name1 = global_var['nom'] # New result
        #print tb_old_name

        if np.all(tb_old_name != name1) and global_var['flag_recog']: # if new result is different to old results
            print 'Essaie ' + str(nb_time+1) + ': reconnu comme ' + str(name1)
            if (a==0):
                a = 1 # Small trick to not to ask twice (not start two Speech Recognizer) at the same time
                resp = validate_recognition(clientId, name1)
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

        elif (not global_var['flag_recog']):
            print 'Essaie ' + str(nb_time+1) + ': personne inconnue'
            a = 0
            result = 0
            nb_time += 1
            time.sleep(wait_time)

    if (result==1): # User confirms that the recognition is correct now
        global_var['flag_enable_recog'] = 0
        global_var['flag_reidentify']   = 0
        global_var['flag_wrong_recog']  = 0

        get_face_emotion_api_results(clientId)

        global_var['text'], global_var['text2'], global_var['text3'] = go_to_formation(clientId, xls_filename, name)

        return_to_recog(clientId) # Return to recognition program immediately or 20 seconds before returning

    else: # Two time failed to recognized
        global_var['flag_enable_recog'] = 0 # Disable recognition when two tries have failed
        global_var['flag_reidentify']   = 0
        simple_message(clientId, u'Problème méconnaissable', u'Désolé je vous reconnaît pas, veuillez me donner votre identifiant')

        name = ask_name(clientId, 0)
        if os.path.exists(imgPath+str(name)+".0"+suffix): # Assume that user's face-database exists if the photo 0.png exists
            simple_message(clientId, 'Reprise de photos', 'Bonjour '+ str(name)+', je vous conseille de changer vos photos')
            flag_show_photos = 1
            step_time = 1

            thread_show_photos3 = Thread(target = show_photos, args = (clientId, imgPath, name), name = 'thread_show_photos3_'+clientId)
            thread_show_photos3.start()

            time.sleep(0.5)
            thread_retake_validate_photos2 = Thread(target = retake_validate_photos, args = (clientId, step_time, flag_show_photos, imgPath, name), name = 'thread_retake_validate_photos2_'+clientId)
            thread_retake_validate_photos2.start()
        else:
            simple_message(clientId, 'Erreur', "Malheureusement, les photos correspondant au nom "+ str(name) +" n'existent pas. Je vous conseille de reprendre vos photos")
            global_var['flag_take_photo']  = 1  # Enable photo taking

"""
==============================================================================
Main program body with decision and redirection
==============================================================================
"""
def run_program(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    # Autorisation to begin Streaming Video
    optin0 = allow_streaming_video(clientId)

    if (optin0 == 1):
        global_var['key']       = 0

        # Thread of streaming video
        start_time   = time.time() # For recognition timer (will reset after each 3 secs)
        time_origine = time.time() # For display (unchanged)

        """
        Permanent loop
        """
        while (time.time() - time_origine < 120):
            data = global_var['binary_data']

            """
            Decision part
            """
            elapsed_time = time.time() - start_time
            if ((elapsed_time > wait_time) and global_var['flag_enable_recog']): # Identify after each 3 seconds
                faceDetectResult    = face_api.faceDetect(None, None, data)
                if len(faceDetectResult)>=1:
                    new_faceId          = faceDetectResult[0]['faceId']
                    resultIdentify      = face_api.faceIdentify(groupId, [new_faceId], maxNbOfCandidates)
                    
                    if len(resultIdentify[0]['candidates'])>=1: # If the number of times recognized is big enough
                        global_var['flag_recog']  = 1 # Known Person
                        global_var['flag_ask']    = 0
                        recognizedPersonId  = resultIdentify[0]['candidates'][0]['personId']
                        recognizedPerson    = face_api.getPerson(groupId, recognizedPersonId)
                        recognizedPerson    = recognizedPerson.replace('null','None')
                        recognizedPerson    = eval(recognizedPerson)
                        global_var['nom']   = recognizedPerson['name']
                        global_var['text']  = 'Reconnu : ' + global_var['nom']

                        print global_var['text']

                        if (not global_var['flag_reidentify']):
                            global_var['text2'] = "Appuyez [Y] si c'est bien vous"
                            global_var['text3'] = "Appuyez [N] si ce n'est pas vous"

                            res_verify_recog = verify_recog(clientId, global_var['nom'])
                            if (res_verify_recog==1):
                                global_var['key'] = ord('y')
                            elif (res_verify_recog==0):
                                global_var['key'] = ord('n')

                    else: # If the number of times recognized anyone from database is too low
                        global_var['flag_recog'] = 0 # Unknown Person
                        global_var['text']  = 'Personne inconnue'
                        global_var['text2'] = ''
                        global_var['text3'] = ''

                        if (not global_var['flag_reidentify']):
                            global_var['flag_ask'] = 1
                            simple_message(clientId, '', u'Désolé, je ne vous reconnaît pas')
                else:
                    global_var['flag_recog'] = -1
                    global_var['text']  = 'Aucune personne'
                    global_var['text2'] = ''
                    global_var['text3'] = ''

                start_time = time.time()  # reset timer

            """
            Redirecting user based on recognition result and user's status (already took photos or not) in database
            """
            count_time = time.time() - time_origine
            if (count_time <= wait_time):
                global_var['text3'] = 'Initialisation (pret dans ' + str(wait_time-count_time)[0:4] + ' secondes)...'

            else:
                """
                Start Redirecting after the first 1.5 seconds
                """

                if (global_var['flag_recog']==1):
                    if (global_var['key']==ord('y') or global_var['key']==ord('Y')): # User chooses Y to go to Formation page
                        global_var['flag_wrong_recog']  = 0
                        get_face_emotion_api_results(clientId)
                        global_var['text'], global_var['text2'], global_var['text3'] = go_to_formation(clientId, xls_filename, global_var['nom'])

                        global_var['key'] = 0
                        return_to_recog(clientId) # Return to recognition program, after displaying Formation

                    if (global_var['key']==ord('n') or global_var['key']==ord('N')): # User confirms that the recognition result is wrong by choosing N
                        global_var['flag_wrong_recog'] = 1
                        global_var['flag_ask'] = 1
                        global_var['key'] = 0


                if ((global_var['flag_recog']==1 and global_var['flag_wrong_recog']==1) or (global_var['flag_recog']==0)): # Not recognized or not correctly recognized
                    if (global_var['flag_ask']==1):# and (not flag_quit)):
                        resp_deja_photos = deja_photos(clientId) # Ask user if he has already had a database of face photos
                        print 'resp_deja_photos = ', resp_deja_photos
                        
                        if (resp_deja_photos==-1):
                            global_var['flag_ask'] = 0

                        elif (resp_deja_photos==1): # User has a database of photos
                            global_var['flag_enable_recog'] = 0 # Disable recognition in order not to recognize while re-identifying
                            global_var['flag_ask'] = 0

                            name0 = global_var['nom']   # Save the recognition result, which is wrong, in order to compare later
                            nb_time_max = 2             # Number of times to retry recognize

                            thread_reidentification = Thread(target = re_identification, args = (clientId, nb_time_max, name0), name = 'thread_reidentification_'+clientId)
                            thread_reidentification.start()

                        elif (resp_deja_photos == 0): # User doesnt have a database of photos

                            global_var['flag_enable_recog'] = 0 # Disable recognition in order not to recognize while taking photos
                            resp_allow_take_photos = allow_take_photos(clientId)

                            if (resp_allow_take_photos==1): # User allows to take photos
                                global_var['flag_take_photo'] = 1  # Enable photo taking

                            else: # User doesnt want to take photos
                                global_var['flag_take_photo'] = 0
                                res = allow_go_to_formation_by_id(clientId)
                                if (res==1): # User agrees to go to Formation in providing his id manually
                                    name = ask_name(clientId, 1)
                                    global_var['text'], global_var['text2'], global_var['text3'] = go_to_formation(clientId, xls_filename, name)

                                    # Return to recognition program (if user wishs to, otherwise, wait 20 seconds before returning anyway)
                                    return_to_recog(clientId)

                                else: # Quit if user refuses to provide manually his id (after all other functionalities)
                                    break

                            resp_allow_take_photos = 0
                        resp_deja_photos = 0
                    global_var['flag_ask'] = 0

                if (global_var['flag_take_photo']==1):# and (not flag_quit)):
                    step_time  = 1 # Interval of time (in second) between two times of taking photo

                    thread_take_photo = Thread(target = take_photos, args = (clientId, step_time, 1), name = 'thread_take_photo_'+clientId)
                    thread_take_photo.start()

                    global_var['flag_take_photo'] = 0

                """
                Call Face API and Emotion API, and display
                """
                if (global_var['key']==ord('i') or global_var['key']==ord('I')):
                    retrieve_face_emotion_att(clientId)
                    global_var['key'] = 0
        """
        End of While-loop
        """
    """
    Exit the program
    """
    quit_program(clientId)


"""
Quit program
"""
def quit_program(clientId):

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()

    global_var['flag_quit'] = 0 # Turn it on to execute just the yes_no question and bye-bye
#    quit_opt = yes_or_no(clientId, 'Exit', 'Voulez-vous vraiment quitter ?', 3) # This wont executed if quit by Esc key
    
    chrome_tts(clientId, u"{} secondes ont été passées".format(str(time.time() - origine_time).split('.')[0]))
    chrome_tts(clientId, u"Merci de votre utilisation. Au revoir, à bientôt")
    global_var['flag_quit'] = 1

"""
Definition of used yes-no questions in program
"""
def detect_face_attributes(clientId):
    resp = yes_or_no(clientId,"", "Voulez-vous apercevoir vos attributs faciales ?", 3)
    return resp

def verify_recog(clientId, name):
    resp = yes_or_no(clientId,"", "Bonjour " + name + ", est-ce bien vous ?", 3)
    return resp

def allow_streaming_video(clientId):
    resp = yes_or_no(clientId, 'Reconnaissance faciale', 'Bonjour ! Voulez-vous lancer la reconnaissance faciale ?', 4)
    return resp

def deja_photos(clientId):
    resp = yes_or_no(clientId, 'Base de photos', u'Avez-vous déjà pris des photos ?', 3)
    return resp

def allow_take_photos(clientId):
    resp = yes_or_no(clientId, 'Prise de photos', u"Êtes-vous d'accord pour vous faire prendre en photos ?", 3)
    return resp

def validate_photo(clientId):
    resp = yes_or_no(clientId, 'Validation de photos', 'Voulez-vous valider ces photos ?', 4)
    return resp

def allow_go_to_formation_by_id(clientId):
    resp = yes_or_no(clientId, 'Accès Formation', u"Voulez-vous accéder votre page Formation par votre identifiant ?", 3)
    return resp

def quit_formation(clientId):
    resp = yes_or_no(clientId, 'Quitter cette page', 'Voulez-vous quitter la page Formation ?', 4)
    return resp

def validate_recognition(clientId, name):
    resp = yes_or_no(clientId,"Re-identification", "Bonjour " + name + ", est-ce bien vous cette fois-ci ?", 4)
    return resp

"""
Yes/No question as an asking/answering by dialogue
"""
def yes_or_no(clientId, title, message, type_message_box):
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    if (not global_var['flag_quit']): # Put in If-condition to stop asking when Esc is pressed
        resp, ouinon = chrome_yes_or_no(clientId, message)
        return resp
    else:
        return -1

"""
Simple message as a notification speech
"""
def simple_message(clientId, title, message):
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    if (not global_var['flag_quit']): # Put in If-condition to allow interrupt when Esc is pressed
        chrome_tts(clientId, message)


"""
Initialisation for global variables used by clientId
"""
def global_var_init(clientId):

    global global_vars

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

    # Initialisation global variables
    key   = 0 # Quit key inside video streaming thread
    key2  = 0 # Quit key from run program

    todo = ''
    stt  = ''
    tts  = ''

    textFromHTML  = ""
    binary_data   = 0

    nom = ''

    global_vars.append( dict([  ('clientId', str(clientId)),
                                ('text', text), ('text2', text2), ('text3', text3),
                                ('age', age), ('gender', gender), ('emo', emo),
                                ('flag_recog', flag_recog),
                                ('flag_take_photo', flag_take_photo),
                                ('flag_wrong_recog', flag_wrong_recog),
                                ('flag_enable_recog', flag_enable_recog),
                                ('flag_disable_detection', flag_disable_detection),
                                ('flag_quit', flag_quit),
                                ('flag_ask', flag_ask),
                                ('flag_reidentify', flag_reidentify),
                                ('key', key),
                                ('key2', key2),
                                ('todo', todo),
                                ('stt', stt),
                                ('tts', tts),
                                ('textFromHTML', textFromHTML),
                                ('binary_data', binary_data),
                                ('nom', nom)
                                ]))


"""
==============================================================================
Flask Initialization
==============================================================================
"""
app  = Flask(__name__)
port = int(os.getenv("PORT", 9099))
 
@app.route('/')
def render_hmtl():
    return render_template('index.html')

@app.route('/start/<clientId>', methods=['POST'])
def onStart(clientId):
    global_var_init(clientId)
    #time.sleep(0.5)
    thread_program = Thread(target = run_program, args= (clientId,), name = 'thread_prog_'+clientId)
    thread_program.start()

    return "", 200

@app.route('/StT/<data>', methods=['POST'])
def runSpeechToText(data):
    clientId = data[0:5]
    text     = data[6:]

    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    global_var['stt'] = text
    #print 'StT', clientId, text
    return "", 200

@app.route('/textFromHTML/<data>', methods=['POST'])
def getTextFromHTML(data):
    clientId = data[0:5]
    text     = data[6:]
    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    global_var['textFromHTML'] = text
    #print 'textFromHTML', clientId, text
    return "", 200

@app.route('/longpolling/<clientId>', methods=['POST'])
def longPolling(clientId):
    time.sleep(0.5)
    global global_vars
    global_var  = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    temp        = global_var['todo']
    global_var['todo'] = ""
    return temp, 200

@app.route('/image', methods=['POST'])
def getImage():
    clientId        = request.get_json(force=True)["client_id"]
    image           = request.get_json(force=True)["img"]
    image           = image.split(",")[1]
    binary_data     = a2b_base64(image)

    global global_vars
    global_var  = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    global_var['binary_data']   = binary_data
    return "",200



"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
# Parameters
root_path    = ""
imgPath      = "face_database_for_oxford/" # path to database of faces
suffix       = '.jpg' # image file extention
thres        = 80     # Distance threshold for recognition
wait_time    = 1.5    # Time needed to wait for recognition
nb_img_max   = 2      # Number of photos needs to be taken for each user
xls_filename = 'formation' # Excel file contains Formation information
maxNbOfCandidates = 1 # Maximum number of candidates for the identification

# Natural Language Classifier
natural_language_classifier = NaturalLanguageClassifierV1(
                              username = '82376208-a089-464c-a5da-96893ed1aa89',
                              password = 'SEuX8ielPiiJ')

# Training Phase
groupId     = "group_all"
groupName   = "employeurs"

face_api.deletePersonGroup(groupId)

# Create PersonGroup
result = face_api.createPersonGroup(groupId, groupName, "")
print result
#result = eval(result)

# if result["error"]["code"] == "PersonGroupExists":
#     #TODO: change to get personGroup info than delete it
#     #face_api.deletePersonGroup(groupId)
#     print 'a'

list_nom = []
list_personId = []
nbr = 0

image_paths = [os.path.join(imgPath, f) for f in os.listdir(imgPath)]

for image_path in image_paths:
    nom = os.path.split(image_path)[1].split(".")[0]
    if nom not in list_nom:
        # Create a Person in a PersonGroup
        personName = nom
        personId   = face_api.createPerson(groupId, personName, "")
        
        list_nom.append(nom)
        list_personId.append(personId)
        nbr += 1
    else:
        personId = list_personId[nbr-1]

    print "Add image...", nom, '\t', image_path
    face_api.addPersonFace(groupId, personId, "", image_path, None)
    time.sleep(0.5)

resultTrainPersonGroup = face_api.trainPersonGroup(groupId)

res      = face_api.getPersonGroupTrainingStatus(groupId)
res      = res.replace('null','None')
res_dict = eval(res)
training_status = res_dict['status']
print training_status


while (training_status=='running'):
    time.sleep(0.5)
    res = face_api.getPersonGroupTrainingStatus(groupId)
    res = res.replace('null','None')
    res_dict = eval(res)
    training_status = res_dict['status']
    print training_status

if (training_status=='succeeded'):
    global_vars = []
    origine_time = time.time()
    app.run(host = '0.0.0.0', port = port, threaded = True)



# END OF PROGRAM
