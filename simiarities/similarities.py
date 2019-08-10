import datetime
from gensim.models.doc2vec import Doc2Vec
from training.training_prefix import makeTrainingPrefix

MODEL_CACHE_TTL_SEC=60*9
MIN_FREE_MEMORY_FOR_CACHE_MB=64

def get_memory_usage():
  """
  Get node total memory and memory usage
  """
  with open('/proc/meminfo', 'r') as mem:
      ret = {}
      tmp = 0
      for i in mem:
          sline = i.split()
          if str(sline[0]) == 'MemTotal:':
              ret['total'] = int(sline[1])
          elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
              tmp += int(sline[1])
      ret['free'] = tmp
      ret['used'] = int(ret['total']) - int(ret['free'])
  return ret

class PostSimilarity:
  modelCache = {}
  modelCachedSavedAt = datetime.datetime.now()

  def getSimilarContentPost(self, text, language, object):
    print("getPostSimilarities")
    print(object)
    print(text)
    print(language)
    filename_prefix = makeTrainingPrefix(language, "posts", object)
    filename = "d2v_models/"+filename_prefix+'_d2v.model'
    print(filename)
    timeNow = datetime.datetime.now()
    timeDifference = (timeNow-PostSimilarity.modelCachedSavedAt).total_seconds()
    print(timeDifference)
    if (PostSimilarity.modelCache.get(filename)==None or timeDifference>MODEL_CACHE_TTL_SEC):
      PostSimilarity.modelCache[filename]=Doc2Vec.load(filename)
      PostSimilarity.modelCachedSavedAt=datetime.datetime.now()
    else:
      print("USING MODEL CACHE")
    self.model = self.modelCache[filename]

    free_memory_mb = get_memory_usage()['free']/1024
    print("Free memory: "+str(free_memory_mb)+"mb")
    if (free_memory_mb<MIN_FREE_MEMORY_FOR_CACHE_MB):
      del PostSimilarity.modelCache[filename]
      PostSimilarity.modelCache[filename]=None
      print("Warning: not enough memory free, deleting cache")

    test_vector = self.model.infer_vector([text],
                alpha = 0.025,
                steps= 20,
                min_alpha = 0.00025,
                epochs = 100)
    most_similar = (self.model.docvecs.most_similar(positive=[test_vector], topn = 5))
    return most_similar
