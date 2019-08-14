from gensim.models import Word2Vec
from reynir import Reynir
import regex as re

# Yields all lemmas in every sentences
# Structure: [file[sent[lemmas]]]
def getLemmatizedTextIS(text):
  lemmas = ' '.join(lemmatizeParse(text))
  return lemmas

def lemmatizeParse(text):
  # For parsing sentences
  reynir = Reynir()
  lemmas = []
  sents = reynir.parse(text)
  for sent in sents['sentences']:
      try:
          lemmas.append(' '.join(sent.tree.lemmas))
      except AttributeError:
          print("ERROR: lemmatize AttributeError, adding raw: "+str(sent))
          lemmas.append(str(sent))
          pass
  return lemmas


