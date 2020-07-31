import re
from lemmatizer.lemmatizer_xx import LemmatizerMultilanguageClass
from lemmatizer.lemmatizer_is_reynir_hybrid import getLemmatizedTextIS
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_short

lemmatizerMultilanguage = LemmatizerMultilanguageClass()

#TODO: Use spacy where possible
#TODO: Use SnowballStemmer for whats there

def getLemmatizedText(name, content, language):
  language = language[:2]
  language = language.lower()
  outText = ""
  if (language):
    if (language=="is"):
      outText = getLemmatizedTextIS(name, content)
      print("IS")
    else:
      outText = lemmatizerMultilanguage.getLemmatizedText(language, name+" "+content)
      print(language.upper())
  else:
    text = name+" "+content
    outText = text.lower().replace('.','.')
    print("ERROR: No language for Lemmatizing text")
  cleaned = re.sub(' +', ' ',outText)
  cleaned = cleaned.replace('\n', '')
  cleaned = cleaned.replace('\r', '')

  cleaned = remove_stopwords(cleaned)
  cleaned = strip_tags(cleaned)
  cleaned = strip_punctuation(cleaned)
  cleaned = strip_numeric(cleaned)
  cleaned = strip_short(cleaned, 1)
  cleaned = strip_multiple_whitespaces(cleaned)
  cleaned = cleaned.lower()

  print("Lemmatized CLEAN: "+cleaned)
  return cleaned
