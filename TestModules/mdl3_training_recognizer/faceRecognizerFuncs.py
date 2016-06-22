# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:39:33 2016

@author: GGQN0871
"""

def setLabelsInfo(filename, labelInfoDict):
    f = open(filename, "r")
    text = f.read()
    f.close()
    
    begin = text.find('<labelsInfo>')
    end = text.find('</labelsInfo>')
    
    labelInfoDict = str(labelInfoDict)
    labelInfoDict = labelInfoDict.replace("'", "x")
    labelInfoDict = labelInfoDict.replace('{', '"&lt;')
    labelInfoDict = labelInfoDict.replace('}', '&gt;"')

    text2 = text[0:begin+len('<labelsInfo>')] + labelInfoDict + text[end:]
    f2 = open(filename, "w")
    f2.write(text2)
    f2.close()
    
def getLabelsInfo(filename):
    f = open(filename, "r")
    text = f.read()
    f.close()
    
    begin = text.find('<labelsInfo>')
    end = text.find('</labelsInfo>')
    
    labelInfoDict = text[begin+len('<labelsInfo>'):end]
    labelInfoDict = labelInfoDict.replace("x", "'")
    labelInfoDict = labelInfoDict.replace('"&lt;','{')
    labelInfoDict = labelInfoDict.replace('&gt;"','}') 
    
    return eval(labelInfoDict)


#filename = 'faceRecognizer_test.xml'
#labelInfoDict = dict([('0', 'a'), ('1', 'b')])
#text2 = setLabelsInfo(filename, labelInfoDict)
#
#labelsInfoDict = getLabelsInfo(filename)
