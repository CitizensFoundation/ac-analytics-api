import re
from lemmatizer.lemmatizer_en import getLemmatizedTextEN
from lemmatizer.lemmatizer_is import getLemmatizedTextIS

def getLemmatizedText(text, language):
  outText = ""
  text = text.lower()
  if (language):
    if (language=="en"):
      outText = getLemmatizedTextEN(text)
      #print("Lemmatized EN: "+outText)
    elif (language=="is"):
      outText = getLemmatizedTextIS(text)
      #print("Lemmatized IS: "+outText)
    else:
      outText = text
      print("WARNING: Could not find Lemmatizer for language: "+language+" text "+text)
  else:
    outText = text
    print("ERROR: No language for Lemmatizing text: "+text)
  cleaned = re.sub(' +', ' ',outText)
  cleaned = cleaned.replace('\n', '')
  cleaned = cleaned.replace('\r', '')
  print("Lemmatized: "+cleaned)
  return cleaned
