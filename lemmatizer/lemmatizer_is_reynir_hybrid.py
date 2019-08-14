from gensim.models import Word2Vec
from reynir.bincompress import BIN_Compressed
from reynir import Reynir
import regex as re
from nltk.tokenize import sent_tokenize, word_tokenize

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
          bin = BIN_Compressed()
          bin_lemmas = []
          sent_words = sent_tokenize(sent.lower())
          sent_words = word_tokenize(' '.join(sent_words))
          for word in sent_words:
            word_lookup = bin.lookup(word)
            if word_lookup!= []:
              bin_lemmas.append(word_lookup[0][0])
            elif bin.lookup(word) == []:
              bin_lemmas.append(word)

          lemmas.append(' '.join(bin_lemmas))
          pass
  return lemmas


