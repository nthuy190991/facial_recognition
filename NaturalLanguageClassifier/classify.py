import json, sys
from watson_developer_cloud import NaturalLanguageClassifierV1

natural_language_classifier = NaturalLanguageClassifierV1(
  username='82376208-a089-464c-a5da-96893ed1aa89',
  password='SEuX8ielPiiJ')

try:
    text = sys.argv[1]
except IndexError:
    print "No parameter !"
    text = 'ok, on fait comme ca !'

classes = natural_language_classifier.classify('2374f9x68-nlc-1265', text)
print classes["top_class"]
#print(json.dumps(classes, indent=2))
