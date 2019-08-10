from training.training_manager import TrainingManager
from training.create_weights import CreateWeights

def triggerPostTraining(type, object):
  print("triggerPostTraining")
  trainer = TrainingManager("posts","post",object)
  trainer.start()
  weights = CreateWeights("posts","post",object,trainer.model)
  weights.start()

def triggerPointTraining(type, object):
  print("triggerPointTraining")
  trainer = TrainingManager("points","post",object)
  trainer.start()
  weights = CreateWeights("points","post",object,trainer.model)
  weights.start()

def triggerArticleTraining(type, object):
  print("triggerArticleTraining")
  trainer = TrainingManager("articles","article",object)
  trainer.start()
  weights = CreateWeights("articles","article",object,trainer.model)
  weights.start()
