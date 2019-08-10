from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
import regex as re
import os

class TrainDoc2Vec:
  def __init__(self,
               filename_prefix,
               textArray,
               idArray,
               vector_size = 300,
               alpha = 0.025,
               min_alpha = 0.00025,
               min_count = 1,
               epochs = 100,
               dm = 0):
    self.filename_prefix = filename_prefix
    self.textArray = textArray
    self.idArray = idArray
    self.vector_size = vector_size
    self.alpha = alpha
    self.min_alpha = min_alpha
    self.min_count = min_count
    self.epochs = epochs
    self.dm = dm

  def train(self):
    # All filenames as a separate list for tagging with TaggedDocument
    # Structure: [[file1], [file2]]
    # (I honestly don't know why Doc2Vec only accepts this format as the input)
    all_ids = [self.idArray[i:i+1] for i in range(0, len(self.idArray))]

    #TODO: Make this work
    #self.textArray = [w for w in self.textArray if not w in stop_words]  # Remove stopwords.
    #self.textArray = [w for w in self.textArray if w.isalpha()]
    #print(self.textArray)
    # Input for build_vocab
    tagged_data = [TaggedDocument(words = word_tokenize(d.lower()),
                  tags = all_ids[i]) for i, d in enumerate(self.textArray)]
    #print(tagged_data)

    self.model = Doc2Vec(vector_size = self.vector_size,
                    alpha = self.alpha,
                    min_alpha = self.min_alpha,
                    min_count = self.min_count,
                    epochs = self.epochs,
                    dm = self.dm)

    self.model.build_vocab(tagged_data)

    try:
      self.model.train(tagged_data,
                  total_examples = self.model.corpus_count,
                  epochs = self.model.epochs)
    except Exception as e:
      print("ERROR in training: "+str(e))
      print(tagged_data)
      return

    if not os.path.exists("d2v_models"):
      os.makedirs("d2v_models")
    filename = "d2v_models/"+self.filename_prefix+'_d2v.model'

    self.model.save(filename)
    print(f"Model ready: {filename}")