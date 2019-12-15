from training.training_manager import TrainingManager
from training.weights_manager import WeightsManager

def triggerPostTraining(type, object):
  print("triggerPostTraining")
  trainer = TrainingManager("posts_"+object["cluster_id"],"post",object)
  trained = trainer.start()
  if trained:
    weights = WeightsManager("posts_"+object["cluster_id"],"post",object,trainer.model)
    weights.startProcessing()

def triggerPointTraining(type, object):
  print("triggerPointTraining")
  trainer = TrainingManager("points_"+object["cluster_id"],"point",object)
  trained = trainer.start()
  if trained:
    weights = WeightsManager("points_"+object["cluster_id"],"point",object,trainer.model)
    weights.startProcessing()

def triggerArticleTraining(type, object):
  print("triggerArticleTraining")
  trainer = TrainingManager("articles_"+object["cluster_id"],"article",object)
  trained = trainer.start()
  if trained:
    weights = WeightsManager("articles_"+object["cluster_id"],"article",object,trainer.model)
    weights.startProcessing()
