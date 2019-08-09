from gensim.models import Word2Vec
import warnings
import nltk
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

class TrainWord2Vec:
  def __init__(self, filename_prefix, text, vec_min_count=3, vec_size=100, vec_window=5, vec_workers=8):
    self.filename_prefix = filename_prefix
    self.text = text
    self.vec_min_count = vec_min_count
    self.vec_size = vec_size
    self.vec_window = vec_window
    self.vec_workers = vec_workers

  def train(self):
    print("Tokenizing sentences")
    all_sentences = nltk.sent_tokenize(self.text)
    print("All sentences ready")

    print("Tokenizing words")
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    print("All words ready")

    print("Training model")
    model1 = Word2Vec(all_words, min_count = self.vec_min_count,
                      size = self.vec_size, window = self.vec_window, workers = self.vec_workers)

    print("Model trained")
    print("Saving model")
    if not os.path.exists("w2v_models"):
      os.makedirs("w2v_models")
    filename = "w2v_models/"+self.filename_prefix+'_w2v.model'
    model1.save(filename)
    print(f"Model ready: {filename}")
