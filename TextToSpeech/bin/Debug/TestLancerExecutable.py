
import subprocess
#mettre le chemin pour le .exe et eviter que ce soit purement en dur, ça n'est pas gérable sinon
#subprocess.Popen(r"C:\Applications\testexe\version 4.0\tts - stt - azure ML - MSFT\VisualStudioProjects\TalkerPy\TalkerPy\bin\Debug\TalkerPy.exe")
WhatIsToBeSpoken = 'êtes-vous daccord pour vous faire prendre en photo ?'
args = [r'C:\Applications\testexe\version 4.0\tts - stt - azure ML - MSFT\VisualStudioProjects\BinTextToSpeech\TalkerPy.exe', WhatIsToBeSpoken]
subprocess.call(args)
