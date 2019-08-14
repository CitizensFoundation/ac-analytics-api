from training.training_manager import TrainingManager
from training.weights_manager import WeightsManager

def triggerPostTraining(type, object):
  print("triggerPostTraining")
  trainer = TrainingManager("posts","post",object)
  trainer.start()
  weights = WeightsManager("posts","post",object,trainer.model)
  weights.startProcessing()

def triggerPointTraining(type, object):
  print("triggerPointTraining")
  trainer = TrainingManager("points","post",object)
  trainer.start()
  weights = WeightsManager("points","post",object,trainer.model)
  weights.startProcessing()

def triggerArticleTraining(type, object):
  print("triggerArticleTraining")
  trainer = TrainingManager("articles","article",object)
  trainer.start()
  weights = WeightsManager("articles","article",object,trainer.model)
  weights.startProcessing()
