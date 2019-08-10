from gensim.models import Word2Vec
from reynir import Reynir
import regex as re

# For parsing sentences
reynir = Reynir()

# Yields all lemmas in every sentences
# Structure: [file[sent[lemmas]]]
def getLemmatizedTextIS(text):
  lemmas = ''.join(lemmatize(text))
  lemmas = lemmas.replace(" .", ". ")
  lemmas = lemmas.replace(" !", "!")
  return lemmas

def lemmatize(text):
  lemmas = []
  sents = reynir.parse(text)
  for sent in sents['sentences']:
      try:
          reynir_lemmas = sent.tree.lemmas
          lemmas.append(' '.join(reynir_lemmas))
      except AttributeError:
          print("ERROR: lemmatize AttributeError, adding raw: "+text)
          lemmas.append(text)
          pass
  return lemmas


