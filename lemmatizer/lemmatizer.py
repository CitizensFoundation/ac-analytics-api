import re
from lemmatizer.lemmatizer_en import getLemmatizedTextEN
from lemmatizer.lemmatizer_is_reynir_hybrid import getLemmatizedTextIS

#TODO: Use spacy where possible
#TODO: Use SnowballStemmer for whats there

def getLemmatizedText(text, language):
  language = language[:2]
  language = language.lower()
  outText = ""
  if (language):
    if (language=="en"):
      outText = getLemmatizedTextEN(text)
      print("Lemmatized EN: "+outText)
    elif (language=="is"):
      outText = getLemmatizedTextIS(text)
      print("Lemmatized IS: "+outText)
    else:
      outText = text.lower().replace('.','.')
      print("warning: No Lemmatizer for language: "+language+" text "+text)
  else:
    outText = text.lower().replace('.','.')
    print("ERROR: No language for Lemmatizing text: "+text)
  cleaned = re.sub(' +', ' ',outText)
  cleaned = cleaned.replace('\n', '')
  cleaned = cleaned.replace('\r', '')
  print("Lemmatized CLEAN: "+cleaned)
  return cleaned
