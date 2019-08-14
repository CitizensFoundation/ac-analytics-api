from reynir.bincompress import BIN_Compressed
import glob
import regex as re
from nltk.tokenize import sent_tokenize, word_tokenize

def getLemmatizedTextIS(text):
  lemmas = ' '.join(lemmatizeBin(text))
  print(lemmas)
  return lemmas

def lemmatizeBin(txt):
  bin = BIN_Compressed()
  lemmas = []
  txt = sent_tokenize(txt.lower())
  txt = word_tokenize(' '.join(txt))

  for word in txt:
    word_lookup = bin.lookup(word)
    if word_lookup!= []:
      lemmas.append(word_lookup[0][0])
    elif bin.lookup(word) == []:
      lemmas.append(word)

  return lemmas


