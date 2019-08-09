from gensim.models import Word2Vec
from reynir import Reynir
import regex as re

# For parsing sentences
reynir = Reynir()

# Yields all lemmas in every sentences
# Structure: [file[sent[lemmas]]]
def getLemmatizedTextIS(text):
  out = [str(l) for l in lemmatize(text)]
  return ' '.join(out)

def lemmatize(text):
  lemmas = []
  sents = reynir.parse(file)
  for sent in sents['sentences']:
      try:
          lemmas.append(sent.tree.lemmas)
      except AttributeError:
          print("ERROR: lemmatize AttributeError")
          pass
  yield lemmas
