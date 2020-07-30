NUM_THREADS = 8
SAVE_LOAD_TEST = False
import sys
import os
import numpy as np  # linear algebra
sys.path.append(".")
from lightfm.evaluation import auc_score

from datetime import datetime
from lightfm_model_cache import LightFmModelCache
from training_manager import RecTrainingManager
from elasticsearch import Elasticsearch, helpers, exceptions
import json

cluster_id = 1

class RecommendationPrediction:

  def __init__(self, cluster_id):
        self._cluster_id = cluster_id

  def predict_for_post_ids(self, user_id, post_ids, max_number = 10000, only_return_ids = True):
    model, user_id_map, user_features, item_id_map, item_features, interactions = LightFmModelCache.get(self._cluster_id)

    user_x = user_id_map[user_id]

    search_for_item_ids = []
    search_for_post_ids = []

    for post_id in post_ids:
      search_for_item_ids.append(item_id_map[post_id])
      search_for_post_ids.append(post_id)

    predictions = model.predict(user_x, search_for_item_ids)

    i = 0
    results_tuples = []
    for post_id in search_for_post_ids:
      results_tuples.append((post_id, predictions[i]))
      i = i + 1

    results_tuples.sort(key=lambda x:x[1], reverse = True)

    if only_return_ids:
      only_ids = []
      for tuple in results_tuples:
        only_ids.append(int(tuple[0]))
      print(only_ids)
      return only_ids[0:max_number];
    else:
      return results_tuples[0:max_number]

  #TODO: Optimize this and cache the es connection in a class var
  def get_es_post_ids(self, search_terms):
    es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get(
        'AC_SIM_ES_URL') != None else 'localhost:9200'
    es_client = Elasticsearch(es_url)
    resp = helpers.scan(
        es_client,
        query = { "query": {
          "bool": {
            "must": {
                "term": search_terms
              }
            }
          }
        },
        index='posts_'+str(self._cluster_id),
        scroll='3m'
    )

    def get_only_id_strs(n):
        return str(n["_id"])

    result_list = list(resp)
    only_ids = list(map(get_only_id_strs, result_list))
    print(only_ids)

    return only_ids

  def predict_for_collection(self, user_id, search_terms):
    post_ids = self.get_es_post_ids(search_terms)
    return self.predict_for_post_ids(str(user_id), post_ids)

  def predict_for_domain(self, user_id, domain_id):
    return self.predict_for_collection(user_id, { "domain_id": int(domain_id) })

  def predict_for_community(self, user_id, community_id):
    return self.predict_for_collection(user_id, { "community_id": int(community_id) })

  def predict_for_group(self, user_id, group_id):
    return self.predict_for_collection(user_id, { "group_id": int(group_id) })

prediction = RecommendationPrediction(1)
print(prediction.predict_for_community(850,973))

