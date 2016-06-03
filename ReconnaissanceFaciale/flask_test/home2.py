# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 03:43:38 2016

@author: thnguyen
"""
from flask import Flask, render_template
import time

def flask_init():
    global app, todo, stt, textFromHTML
    todo = ""
    stt  = ""
    textFromHTML = ""

    app  = Flask(__name__)
    
    @app.route('/')
    def render_hmtl():
        return render_template('new.html')
        
    @app.route('/start', methods=['POST'])
    def onStart():
        global todo
        todo = "STT"
        return "", 200
        
    @app.route('/StT/<text>', methods=['POST'])
    def SpeechToText(text):
        global stt, todo
        stt = text
        todo = "TTS " + text
        return "", 200
    
    @app.route('/textFromHTML/<text>', methods=['POST'])
    def getTextFromHTML(text):
        global textFromHTML
        textFromHTML = text
        print 'textFromHTML: ', textFromHTML
		
        return "", 200
        
    @app.route('/longpolling', methods=['POST'])
    def LongPolling():
        time.sleep(0.5)
        global todo
        temp = todo
        todo = ""
        return temp, 200
        
if __name__ == '__main__':
    flask_init()
    app.run(host='0.0.0.0', port=9999)