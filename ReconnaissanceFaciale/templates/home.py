from flask import Flask, request, render_template
import time
import base64
from binascii import a2b_base64
import json

app = Flask(__name__)

todo = "test"
TTS = ""

@app.route('/')
def hello_world():
    return render_template('hello.html')

@app.route('/StT/<text>', methods=['POST'])
def SpeechToText(text):
	global TTS
	global todo
	todo = "TTS " + text
	TTS = text
	print TTS
	return "",200

@app.route('/longpolling', methods=['POST'])
def LongPolling():
	time.sleep(0.5)
	temp = todo
	global todo
	todo = ""
	return temp,200
	
@app.route('/image', methods=['POST'])
def GetImage():
	image = request.get_json(force=True)["img"]
	image = image.split(",")[1]
	binary_data = a2b_base64(image)
	fd = open('image.jpeg','wb')
	fd.write(binary_data)
	fd.close()
	return "",200

def prog():
	global todo
	todo = "STT"
	
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
