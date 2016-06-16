# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 13:20:11 2016

@author: thnguyen
"""

from processRequest import processRequest
import httplib, urllib

_key_face   = '5d99eec09a7e4b2a916eba7f75671600' # primary key
_url        = 'https://api.projectoxford.ai/face/v1.0/'

def createPersonGroup(personGroupId, groupName, userData):

#    headers = {
#        'Content-Type': 'application/json',
#        'Ocp-Apim-Subscription-Key': _key_face,
#    }
#    body = {
#        'name': groupName,
#        'userData': userData
#    }
#    #res = requests.request( 'put', _url_person_group+'/'+personGroupId, json = body, headers = headers)
#    url = _url + 'persongroups/' + personGroupId
#    processRequest('put', url, body, None, headers, None, 2 )

    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    params = urllib.urlencode({
        "personGroupId":personGroupId
    })
    body = {
        "name":groupName,
        "userData":userData
    }
    try:
        conn = httplib.HTTPSConnection('api.projectoxford.ai')
        conn.request("PUT", "/face/v1.0/persongroups/{personGroupId}?%s" % params, "%s"%body, headers)
        response = conn.getresponse()
        result = response.read()
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        result = 'Errno {0}'.format(e.errno)
        #print 'error', e
    return result



def trainPersonGroup(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    #params = {}
    params = urllib.urlencode({
        "personGroupId":personGroupId
    })
    body = {}
    try:
        conn = httplib.HTTPSConnection('api.projectoxford.ai')
        #conn.request("POST", "/face/v1.0/persongroups/"+personGroupId+"/train?%s" % params, "{body}", headers)
        conn.request("POST", "/face/v1.0/persongroups/{personGroupId}/train?%s" % params, "%s"%body, headers)
        response = conn.getresponse()
        result = response.read()
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        result = 'Errno {0}'.format(e.errno)
    return result



def deletePersonGroup(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    params = urllib.urlencode({
        "personGroupId":personGroupId
    })
    body = urllib.urlencode({
    })
    try:
        conn = httplib.HTTPSConnection('api.projectoxford.ai')
        conn.request("DELETE", "/face/v1.0/persongroups/{personGroupId}?%s" % params, "%s"%body, headers)
        response = conn.getresponse()
        result = response.read()
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        result = 'Errno {0}'.format(e.errno)
    return result



def getPersonGroupTrainingStatus(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    params = urllib.urlencode({
        "personGroupId":personGroupId
    })
    body = urllib.urlencode({
    })
    try:
        conn = httplib.HTTPSConnection('api.projectoxford.ai')
        conn.request("GET", "/face/v1.0/persongroups/{personGroupId}/training?%s" % params, "%s"%body, headers)
        response = conn.getresponse()
        result = response.read()
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        result = 'Errno {0}'.format(e.errno)
    return result


def getListPersonsInPersonGroup(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    params = urllib.urlencode({
        "personGroupId":personGroupId
    })
    body = urllib.urlencode({
    })
    try:
        conn = httplib.HTTPSConnection('api.projectoxford.ai')
        conn.request("GET", "/face/v1.0/persongroups/{personGroupId}/persons?%s" % params, "%s"%body, headers)
        response = conn.getresponse()
        result = response.read()
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        result = 'Errno {0}'.format(e.errno)
    return result



def createPerson(personGroupId, personName, userData):

    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    body = {
        'name': personName,
        'userData': userData
    }
    url = _url + 'persongroups/' + personGroupId + '/persons'
    result, status_code = processRequest('post', url, body, None, headers, None, 2 )
    return result['personId']



def getPerson(personGroupId, personId):
    headers = {
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    params = urllib.urlencode({
        "personGroupId":personGroupId,
        "personId":personId
    })
    body = urllib.urlencode({
    })
    try:
        conn = httplib.HTTPSConnection('api.projectoxford.ai')
        conn.request("GET", "/face/v1.0/persongroups/{personGroupId}/persons/{personId}?%s" % params, "%s"%body, headers)
        response = conn.getresponse()
        result = response.read()
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))
        result = 'Errno {0}'.format(e.errno)
    return result


def faceDetect(urlImage, pathToImageInDisk, imageData):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
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
    # Face detection parameters
    params = { 'returnFaceAttributes': 'age,gender,glasses',
               'returnFaceLandmarks': 'true'}
    url_face = _url + 'detect'
    result, status_code = processRequest('post', url_face, json, data, headers, params, 10 )
    return result


def addPersonFace(personGroupId, personId, urlImage, pathToImageInDisk, imageData):
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face

    if ((urlImage==None) or (urlImage==""))  and (imageData==None):
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

    url = _url + 'persongroups/' + personGroupId + '/persons/' + personId + '/persistedFaces'
    result, status_code = processRequest('post', url, json, data, headers, None, 2 )
    return result['persistedFaceId']



def faceIdentify(personGroupId, faceIds, nbCandidate):
    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    json = {
        "personGroupId": personGroupId,
        "faceIds": faceIds,
        "maxNumOfCandidatesReturned": nbCandidate
    }
    url = _url + 'identify'
    result, status_code = processRequest('post', url, json, None, headers, None, 2 )
    return result
