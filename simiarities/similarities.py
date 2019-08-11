import datetime
from gensim.models.doc2vec import Doc2Vec
from training.training_prefix import makeTrainingPrefix
from utils.memory import get_memory_usage

MODEL_CACHE_TTL_SEC=60*9
MIN_FREE_MEMORY_FOR_CACHE_MB=64

class PostSimilarity:
  modelCache = {}
  modelCachedSavedAt = datetime.datetime.now()

  def getSimilarContentPost(self, text, language, object):
    print("getSimilarContentPost")
    filename_prefix = makeTrainingPrefix(language, "posts", object)
    filename = "d2v_models/"+filename_prefix+'_d2v.model'
    print(filename)
    timeNow = datetime.datetime.now()
    timeDifference = (timeNow-PostSimilarity.modelCachedSavedAt).total_seconds()
    if (PostSimilarity.modelCache.get(filename)==None or timeDifference>MODEL_CACHE_TTL_SEC):
      PostSimilarity.modelCache[filename]=Doc2Vec.load(filename)
      PostSimilarity.modelCachedSavedAt=datetime.datetime.now()
    else:
      print("USING MODEL CACHE")

    self.model = self.modelCache[filename]

    #TODO: Confirm this is needed https://github.com/RaRe-Technologies/gensim/issues/2260
    #self.model.docvecs.vectors_docs_norm = None
    #self.model.docvecs.init_sims()

    test_vector = self.model.infer_vector([text.lower().split()],
                alpha = 0.025,
                steps= 20,
                min_alpha = 0.00025,
                epochs = 100)
    most_similar = (self.model.docvecs.most_similar(positive=[test_vector], topn = 5))

    free_memory_mb = get_memory_usage()['free']/1024
    print("Free memory: "+str(free_memory_mb)+"mb")
    if (free_memory_mb<MIN_FREE_MEMORY_FOR_CACHE_MB):
      self.model=None
      del PostSimilarity.modelCache[filename]
      PostSimilarity.modelCache[filename]=None
      print("Warning: not enough memory free, deleting cache")

    return most_similar
