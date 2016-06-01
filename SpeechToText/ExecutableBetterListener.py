
import subprocess
import threading
import os
from threading import Thread
import time


ExePath = r'C:\Applications\RecoFacialFinal\SpeechToText\bin\Debug\RecognizerPython.exe'
TxtWPath = r'C:\Applications\RecoFacialFinal\SpeechToText\RecognizedSpeech.txt'
TxtRPath = r'C:\Applications\RecoFacialFinal\SpeechToText\CommandsRecognizer.txt'


def read_txt(filename):
    file = open(filename, "r")
    text = file.readline()
    file.close()
    return text
    
def write_txt(filename, text):
    ManagedToOpenFile = False
    while not ManagedToOpenFile:
        try:
            file = open(filename, "w")
            ManagedToOpenFile = True
        except IOError: #do nothing it s ok#
            print("didn t work it s ok")
    file.write(text)
    file.close()
    return;   

def threaded_talking(exe, txtR , txtW):
    args = [ exe , txtR , txtW ]
    subprocess.call(args)
    pass

thread = Thread(target = threaded_talking, args = ( ExePath , TxtRPath , TxtWPath ) )#pas oublier le r)
thread.start()




#
#raw_input("Press Enter to continue...")
#ManagedToOpenFile = False
#while not ManagedToOpenFile: # la boucle est necessaire pour gerer les fois ou deux prccessus essaient de lire et ecrire le meme txt en meme temps
#    try:
#        f = open( TxtRPath , 'w')
#        ManagedToOpenFile = True
#    except IOError: #do nothing it s ok#
#        print("didn t work it s ok")
#
#f.write( '0\n1') # python will convert \n to os.linesep
#f.close()

raw_input("Press Enter to continue...")
write_txt(TxtWPath, '*')
write_txt(TxtRPath, '0\n1')
s = read_txt(TxtWPath)
i=0
while s=='*':
    time.sleep(0.2)
    i += 1
    s = read_txt(TxtWPath)
    print i, s



raw_input("Press Enter to continue...")
write_txt(TxtRPath, '1')


#there need to be just a small delay after the last talking to end this process, make sure there is that or  2 processes will write different things in the txt at the same time

#code to end the speaking background process

#raw_input("Press Enter to continue...")
#ManagedToOpenFile = False
#while not ManagedToOpenFile:
#    try:
#        f = open( TxtRPath , 'w')
#        ManagedToOpenFile = True
#    except IOError: #do nothing it s ok#
#        print("didn t work it s ok")
#
#f.write('1')
#f.close()


#code to end the speaking background process



