
from spacy import load
from argparse import ArgumentParser
parser = ArgumentParser()
english = load('en_core_web_sm')

def getLemmatizedTextEN(text):
  out = [str(l) for l in lemmatize(text)]
  return ' '.join(out)

def lemmatize(text):
    words = english(text)
    for token in words:
        if token.lemma_ == '-PRON-':
            yield token
        else:
            yield token.lemma_
