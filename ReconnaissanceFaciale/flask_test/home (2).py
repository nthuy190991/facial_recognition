from flask import Flask
from flask import render_template
import time
from threading import Thread
import thread
import cv2

"""
Initialization of flask
"""
#app  = Flask(__name__)
#todo = ""
#stt  = ""
#tts  = ""
#
#@app.route('/')
#def hello_world():
#    return render_template('hello.html')
#
## @app.route('/start', methods=['POST'])
## def start(tts):
#    # global todo
#    # todo = "TTS bonjour " + tts
#    # return "",200
#    
##@app.route('/TTS', methods=['POST'])
##def TextToSpeech(text):
##    global todo
##    todo = "TTS " + text
##    return;
#    
#@app.route('/StT/<text>', methods=['POST'])
#def SpeechToText(text):
#    global stt
#    #global todo
#    #todo = "TTS " + text
#    stt = text
#    #print stt
#    return "", 200
#    
#@app.route('/longpolling', methods=['POST'])
#def LongPolling():
#    time.sleep(0.5)
#    temp = todo
#    global todo
#    todo = ""
#    return temp, 200


def flask_init():
    global app, todo, stt
    
    app  = Flask(__name__)
    todo = ""
    stt  = ""
    
    @app.route('/')
    def hello_world():
        return render_template('hello.html')
    
    # @app.route('/start', methods=['POST'])
    # def start(tts):
        # global todo
        # todo = "TTS bonjour " + tts
        # return "",200
        
    #@app.route('/TTS', methods=['POST'])
    #def TextToSpeech(text):
    #    global todo
    #    todo = "TTS " + text
    #    return;
        
    @app.route('/StT/<text>', methods=['POST'])
    def SpeechToText(text):
        global stt
        #global todo
        #todo = "TTS " + text
        stt = text
        #print stt
        return "", 200
        
    @app.route('/longpolling', methods=['POST'])
    def LongPolling():
        time.sleep(0.5)
        temp = todo
        global todo
        todo = ""
        return temp, 200


    
def flaskThread():
    app.run(host='0.0.0.0', port=5000)

def ask(question):
    global todo
    todo = "TTS " + question
    
def answer():
    global todo
    global stt
    stt  = ""
    todo = "STT"
    while stt=="":
        time.sleep(0.05) # TODO: How to know when the STT is finished
    resp = stt
    return resp
    
def yes_or_no(question):
    ask(question)
    time.sleep(0.05)
    response = answer()
    return response

def video_streaming():
    
    global key
    
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if (ret == True):
            frame = cv2.flip(frame, 1) # Vertically flip frame
            
            key = cv2.waitKey(1)
            if (key == 27):         # wait for ESC key to exit
                cv2.destroyAllWindows()
                break

        cv2.imshow('Video streaming', frame)   
        
    video_capture.release() # Release video capture
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    
#    thread = Thread(target = app.run(host='0.0.0.0', port=5000))
#    thread.start()
    
    flask_init()
    thread.start_new_thread(flaskThread,())
    
    flag_speech = 1
    count = 0    
    key = 0
    
    thread_video = Thread(target = video_streaming)
    thread_video.start()
    todo = "TTS Bonjour !"
    while True:
        if (key == 27): # wait for ESC key to exit
            todo = "TTS Auvoir !"
            break
            
        elif (key == ord('s')):
            todo = "STT"
            #print stt
        elif (key == ord('t')):
            todo = "TTS " + stt
        elif (key == ord('a')):
            resp = yes_or_no("Bonjour comment vas tu ?")
            print resp
            todo = "TTS Vous avez repondu: " + resp
    cv2.destroyAllWindows()
    
  