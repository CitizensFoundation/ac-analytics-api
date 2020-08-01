from gensim.models import Word2Vec
from reynir.bincompress import BIN_Compressed
from reynir import Reynir
import regex as re
from nltk.tokenize import sent_tokenize, word_tokenize

# Yields all lemmas in every sentences
# Structure: [file[sent[lemmas]]]
def getLemmatizedTextIS(name, content):
  lemmas = ' '.join(lemmatizeParse(name))
  lemmas += ' '
  lemmas += ' '.join(lemmatizeParse(content))
  return lemmas

def lemmatizeParse(text):
  # For parsing sentences
  #print(text)
  reynir = Reynir()
  lemmas = []
  sents = reynir.parse(text)
  for sent in sents['sentences']:
#      print(sent)
      try:
          if sent.lemmas==None:
            raise AttributeError()
          else:
            lemmas.append(' '.join(sent.lemmas))
      except AttributeError:
          print("WARNING: lemmatize AttributeError from Reynir", sent)
          #TODO: CHECK HACKY LINE BELOW RVB
          sent = str(sent)
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

