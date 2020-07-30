import sys
from datetime import datetime

sys.path.append(".")

from lightfm.evaluation import auc_score

from lightfm_model_cache import LightFmModelCache
from training_manager import RecTrainingManager

cluster_id = 1

NUM_THREADS = 8

training_manager = RecTrainingManager()
model, user_id_map, user_features, item_id_map, item_features, train_data, test_data = training_manager.train_with_test_data(cluster_id)

test_auc = auc_score(model,
                     train_data,
                     item_features=item_features,
                     user_features=user_features,
                     num_threads=NUM_THREADS).mean()
print('Train set AUC: %s' % test_auc)

test_auc = auc_score(model,
                     test_interactions=test_data,
                     item_features=item_features,
                     user_features=user_features,
                     num_threads=NUM_THREADS).mean()
print('Test set AUC: %s' % test_auc)
