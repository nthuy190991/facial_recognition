"""
Created on Sat Jun 11 10:16:00 2016

@author: thnguyen
"""
from processRequest import processRequest

_url_emo    = 'https://api.projectoxford.ai/emotion/v1.0/recognize'
_key_emo    = "b226d933ab854505b9b9877cf2f4ff7c" # primary key
_maxNumRetries = 10

def recognizeEmotion(urlImage, pathToImageInDisk, imageData):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_emo
    if ((urlImage==None) or (urlImage=="")) and (imageData==None):
        headers['Content-Type'] = 'application/octet-stream'
        json = None
        with open( pathToImageInDisk, 'rb' ) as f:
            data = f.read()

    if ((pathToImageInDisk==None) or (pathToImageInDisk=="")) and (imageData==None):
        headers['Content-Type'] = 'application/json'
        json = { 'url': urlImage }
        data = None

    if (imageData!=None):
        headers['Content-Type'] = 'application/octet-stream'
        data = imageData
        json = None

    result, status_code = processRequest('post', _url_emo, json, data, headers, None, _maxNumRetries )
    return result
