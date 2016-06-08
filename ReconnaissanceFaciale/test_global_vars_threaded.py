# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 17:55:19 2016

@author: GGQN0871
"""
import time
from threading import Thread 
import cv2

def global_var_init(clientId):
    global global_vars
    
    nom = ''
    global_vars.append(dict([('clientId', str(clientId)), ('nom', nom)]))


def funct(clientId):
    
    global global_vars
    global_var = (item for item in global_vars if item["clientId"] == str(clientId)).next()
    t0 = time.time()
    while (time.time()-t0<=10+clientId):
        time.sleep(0.25)
        #try:
        global_var['nom'] = str(clientId)+'\t'+str(time.time()-t0)+'\n'
        #print global_var['nom']
        print 'thread funct, clientId ' + str(clientId)+'\n'
        thread3 = Thread(target=streaming, args=(clientId,), name=str(clientId))
        thread3.start()
        #except:
         #   print("Unexpected error:", sys.exc_info()[0])
    print 'End Thread clientId: '+str(clientId)+'\n'


def streaming(clientId):
    global i
    #video_capture = cv2.VideoCapture(0)
    t0 = time.time()
    while True:
        #ret, frame = video_capture.read()
        #if (ret == True):
            #frame  = cv2.flip(frame, 1) # Vertically flip frame
        key = cv2.waitKey(1)
        if (key == 27) or (time.time()-t0>5+clientId):         # wait for ESC key to exit
            break
        frame = cv2.imread('a'+str(clientId)+'.jpg')
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time.sleep(0.25)
        print 'thread streaming, clientId ' + str(clientId)+'\n'
        #if (i==0):
        cv2.imshow(str(clientId), frame)   
            #i=1
    #video_capture.release() # Release video capture
    cv2.destroyWindow(str(clientId))


global_vars = []                                
global_var_init(0)
global_var_init(1)

thread1 = Thread(target=funct, args=(0,))
thread2 = Thread(target=funct, args=(1,))

thread1.start()
thread2.start()

print 'Start'
t0=time.time()
while True:
    time.sleep(0.25)
    print str(time.time()-t0)
    time.sleep(0.25)
    print 'thread1.is_alive = ' + str(thread1.is_alive())
    print 'thread2.is_alive = ' + str(thread2.is_alive())
    
    if (time.time()-t0>=20):
        break
    cv2.useOptimized()

    
#thread1 = Thread(target=streaming, args=(0,))
#thread1.start()
#
#thread2 = Thread(target=streaming, args=(1,))
#thread2.start()
    