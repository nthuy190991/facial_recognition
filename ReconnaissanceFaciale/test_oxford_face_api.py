# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 09:37:50 2016

@author: thnguyen
"""

from processRequest import processRequest
import httplib

_key_face   = '5d99eec09a7e4b2a916eba7f75671600' # primary key
_url        = 'https://api.projectoxford.ai/face/v1.0/'
    
def createPersonGroup(personGroupId, groupName, userData):
    
    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    body = {
        'name': groupName,
        'userData': userData
    }
    #res = requests.request( 'put', _url_person_group+'/'+personGroupId, json = body, headers = headers)
    url = _url + 'persongroups/' + personGroupId
    processRequest('put', url, body, None, headers, None, 2 )
        
        
        
        
def trainPersonGroup(personGroupId):
    headers = {
        'Ocp-Apim-Subscription-Key': _key_face,
    }
#    json={}
#    url = _url + 'persongroups/' + personGroupId + '/train'
#    processRequest('post', url, json, None, headers, None, 2 )
    params = {}
    conn = httplib.HTTPSConnection('api.projectoxford.ai')
    conn.request("POST", "/face/v1.0/persongroups/"+personGroupId+"/train?%s" % params, "{body}", headers)
    response = conn.getresponse()
    data = response.read()
    return data
    
    
        
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
    result = processRequest('post', url, body, None, headers, None, 2 )
    return result['personId']




def faceDetect(urlImage):
#    pathToFileInDisk = r'face_database/huy.0.png'
#    with open( pathToFileInDisk, 'rb' ) as f:
#        data = f.read()
        
    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = _key_face
    headers['Content-Type'] = 'application/json' 
    
    # URL direction to image
    json = { 'url': urlImage } 
    data = None

    # Face detection parameters
    params = { 'returnFaceAttributes': 'age,gender', 
               'returnFaceLandmarks': 'true'} 
    url_face = _url + 'detect'
    faceResult = processRequest('post', url_face, json, data, headers, params, 10 )
    return faceResult#[0]['faceId']
    
    
def addPersonFace(personGroupId, personId, urlImage):
    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': _key_face,
    }
    json = { 'url': urlImage } 
    
    url = _url + 'persongroups/' + personGroupId + '/persons/' + personId + '/persistedFaces'
    result = processRequest('post', url, json, None, headers, None, 2 )
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
    result = processRequest('post', url, json, None, headers, None, 2 )
    return result
    

    
        
if __name__ == "__main__":
    
    groupId = "group_id_new"
    groupName = "celebrity"
    
    createPersonGroup(groupId, groupName, "")
    
    personName = "leo1"
    personId = createPerson(groupId, personName, "person 1")
    print personId
    
    
    urlImage = 'http://static.naturallycurly.com/wp-content/uploads/2013/11/rby-hot-facial-hair-leonardo-dicaprio-lgn.jpg'

    faceResult = faceDetect(urlImage)
    persistedFaceId = addPersonFace(groupId, personId, urlImage)
    print persistedFaceId
    
    resultTrainPersonGroup = trainPersonGroup(groupId)
    
    new_faceId = faceDetect('http://www4.pictures.gi.zimbio.com/Cannes+Leonardo+Di+Caprio+Photocall+BHwgyYKvaYil.jpg')
    resultIdentify = faceIdentify(groupId, [new_faceId], 2)
    