from training.training_manager import TrainingManager
from training.create_weights import CreateWeights

def triggerPostTraining(type, object):
  print("triggerPostTraining")
  print(object)
  trainer = TrainingManager("posts","post",object)
  trainer.start()
  weights = CreateWeights("posts","post",object)
  weights.start()

def triggerPointTraining(type, object):
  print("triggerPointTraining")
  print(object)
  trainer = TrainingManager("points","post",object)
  trainer.start()
  weights = CreateWeights("points","post",object)
  weights.start()

def triggerArticleTraining(type, object):
  print("triggerArticleTraining")
  print(object)
  trainer = TrainingManager("articles","article",object)
  trainer.start()
  weights = CreateWeights("articles","article",object)
  weights.start()
