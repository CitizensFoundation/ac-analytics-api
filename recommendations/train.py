NUM_THREADS = 8
SAVE_LOAD_TEST = False
import sys

sys.path.append(".")
from lightfm.evaluation import auc_score

from datetime import datetime
from recommendations.lightfm_model_cache import LightFmModelCache
from recommendations.training_manager import RecTrainingManager

cluster_id = int(sys.argv[1])

training_manager = RecTrainingManager()
model, user_id_map, user_features, item_id_map, item_features, interactions, user_feature_map = training_manager.train(cluster_id)

LightFmModelCache.save(model, user_id_map, user_features, item_id_map, item_features, interactions, user_feature_map, 1)

if SAVE_LOAD_TEST:
  loaded_model, user_id_map, loaded_user_features, item_id_map, loaded_item_features = LightFmModelCache.get(cluster_id)

  test_auc = auc_score(loaded_model,
                      interactions,
                      item_features=loaded_item_features,
                      user_features=loaded_user_features,
                      num_threads=NUM_THREADS).mean()
  print('Train set AUC: %s' % test_auc)