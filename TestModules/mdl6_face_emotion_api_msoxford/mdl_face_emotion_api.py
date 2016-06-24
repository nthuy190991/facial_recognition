import sys
import operator
import face_api
import emotion_api

"""
==============================================================================
Face and Emotion API
==============================================================================
"""
def retrieve_face_emotion_att(filename):

    # Face API
    faceResult = face_api.faceDetect(None, filename, None)

    # Emotion API
    emoResult = emotion_api.recognizeEmotion(None, filename, None)

    # Results
    print 'Found {} '.format(len(faceResult)) + ('faces' if len(faceResult)!=1 else 'face')
    nb_faces     = len(faceResult)
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

        print 'Face index:', '\t', 'Age', '\t', 'Gender', '\t', 'Glasses', '\t', 'Emotion'
        for ind in range(nb_faces):
            print 'Face '+str(ind)+': ', '\t', tb_age[ind], '\t', tb_gender[ind], '\t', tb_glasses[ind], '\t', tb_emo[ind]
    return tb_age, tb_gender, tb_glasses, tb_emo    


"""
==============================================================================
    MAIN PROGRAM
==============================================================================
"""
try:
    imgTest = sys.argv[1]
    if (imgTest == '-h') or (imgTest == '--help'):
        print 'mdl_face_emotion_api.py <path_to_image_test>'
        sys.exit()       
        
except IndexError:
    print "Set parameters as default"
    imgTest = 'test.png'

retrieve_face_emotion_att(imgTest)
