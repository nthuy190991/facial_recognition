# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:30:27 2016

@author: GGQN0871
"""

"""
Replace French accents in text
"""
import ctypes

def replace_accents(text):
    text_origine = ['a','à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ò', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü']
    text_replace  = ['0','\xE0', '\xE1', '\xE2', '\xE3', '\xE4', '\xE5', '\xE6', '\xE7', '\xE8', '\xE9', '\xEA', '\xEB', '\xEC', '\xED', '\xEE', '\xEF', '\xF2', '\xF3', '\xF4', '\xF5', '\xF6', '\xF9', '\xFA', '\xFB', '\xFC']
    for i in range(len(text_origine)):
        print i, text_origine[i], text_replace[i]
        text2 = text.replace(text_origine[i], text_replace[i])
        text  = text2
    return text2


"""
Message Box in Python
"""
def Mbox(title, text, style):
    title2 = replace_accents(title)
    text2  = replace_accents(text)
    ##  Styles:
    ##  0 : OK
    ##  1 : OK | Cancel
    ##  2 : Abort | Retry | Ignore
    ##  3 : Yes | No | Cancel
    ##  4 : Yes | No
    ##  5 : Retry | No
    ##  6 : Cancel | Try Again | Continue
    result = ctypes.windll.user32.MessageBoxA(0, text2, title2, style)

    ## Responses:
    # OK:       return=1
    # Cancel:   return=2
    # Yes:      return=6
    # No:       return=7
    return result

Mbox('',"abàbzehzoêt",0)