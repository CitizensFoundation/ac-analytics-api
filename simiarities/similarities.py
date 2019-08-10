from gensim.models.doc2vec import Doc2Vec
from training.training_prefix import makeTrainingPrefix

def getSimilarContentPost(text, language, object):
  print("getPostSimilarities")
  print(object)
  print(text)
  print(language)
  filename_prefix = makeTrainingPrefix(language, "posts", object)
  filename = "d2v_models/"+filename_prefix+'_d2v.model'
  print(filename)
  model = Doc2Vec.load(filename)
  test_vector = model.infer_vector([text],
               alpha = 0.025,
               steps= 20,
               min_alpha = 0.00025,
               epochs = 100)
  print(test_vector)
  most_similar = (model.docvecs.most_similar(positive=[test_vector], topn = 5))
  return most_similar
