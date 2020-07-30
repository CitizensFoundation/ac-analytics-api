import sys

sys.path.append(".")

from lightfm_model_cache import LightFmModelCache
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset

from training_manager import RecTrainingManager

cluster_id = 1

training_manager = RecTrainingManager()
model, user_id_map, user_features, item_id_map, item_features = training_manager.train(cluster_id)

print("After fit =", datetime.now().strftime("%H:%M:%S"))

LightFmModelCache.save_model(model, user_id_map, user_features, item_id_map, item_features, 1)
