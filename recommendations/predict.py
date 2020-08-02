NUM_THREADS = 4
SAVE_LOAD_TEST = False

import dateutil.parser
import sys
import os
import numpy as np  # linear algebra
from user_agents import parse
import dateutil.parser

sys.path.append(".")

from lightfm.evaluation import auc_score

from datetime import datetime
from recommendations.lightfm_model_cache import LightFmModelCache
from recommendations.training_manager import RecTrainingManager
from elasticsearch import Elasticsearch, helpers, exceptions
import json
from scipy import sparse

cluster_id = 1

class RecommendationPrediction:

  def __init__(self, cluster_id, user_data):
        self._cluster_id = cluster_id
        self._user_data = user_data

  #TODO: Make this code shareable with training_manager
  def new_user_feature_list(self):
    features_array = []
    is_mobile = 2
    is_tablet = 2
    is_pc = 2
    is_bot = 2

    browser_family = "0"

    os_family = "0"

    device_family =  "0"
    device_brand =  "0"

    if 'user_agent' in self._user_data:
        user_agent = parse(self._user_data['user_agent'])
        is_mobile = user_agent.is_mobile
        is_tablet = user_agent.is_tablet
        is_pc = user_agent.is_pc
        is_bot = user_agent.is_bot

        os_family = user_agent.os.family

        browser_family = user_agent.browser.family

        device_family = user_agent.device.family
        device_brand = user_agent.device.brand

    features_array.append("is_mobile:"+str(is_mobile))
    features_array.append("is_tablet:"+str(is_tablet))
    features_array.append("is_pc:"+str(is_pc))
    features_array.append("is_bot:"+str(is_bot))

    features_array.append("browser_family:"+(browser_family if browser_family else "0"))

    features_array.append("os_family:"+(os_family if os_family else "0"))

    features_array.append("device_family:"+(device_family if device_family else ""))
    features_array.append("device_brand:"+(device_brand if device_brand else "0"))

    return features_array


  def format_newuser_input(self, user_features, user_feature_list):
    num_features = len(user_feature_list)
    normalised_val = 1.0
    target_indices = []
    for feature in user_feature_list:
      try:
          print(feature)
          target_indices.append(user_features[feature])
      except KeyError:
          print("new user feature encountered '{}'".format(feature))
          pass

    new_user_features = np.zeros(len(user_features.keys()))
    for i in target_indices:
      new_user_features[i] = normalised_val
    new_user_features = sparse.csr_matrix(new_user_features)
    return(new_user_features)

  def predict_for_post_ids(self, user_id, post_ids, max_number = 10000, only_return_ids = True):
    model, user_id_map, user_features, item_id_map, item_features, interactions, user_features_map = LightFmModelCache.get(self._cluster_id)

    search_for_item_ids = []
    search_for_post_ids = []

    for post_id in post_ids:
      if post_id in item_id_map:
        search_for_item_ids.append(item_id_map[post_id])
        search_for_post_ids.append(post_id)

    print("User id", user_id)

    if len(search_for_item_ids)>0:
      if user_id=="-1" or user_id not in user_id_map:
        print("New user")
        new_user_features = self.format_newuser_input(user_features_map, self.new_user_feature_list())
        predictions = model.predict(0, search_for_item_ids, user_features=new_user_features)
      else:
        print("Known user")
        user_x = user_id_map[user_id]
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
  #      print(only_ids)
        print(len(only_ids))
        return only_ids[0:max_number];
      else:
        return results_tuples[0:max_number]
    else:
      return []

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

    result_list = list(resp)

    final_list = []

    if ('date_options' in self._user_data) and self._user_data['date_options']!=None:
      after_date = dateutil.parser.isoparse(eval(self._user_data['date_options'])['after'])
      print(after_date)

      for post in result_list:
        if (('created_at' in post['_source']) and (post["_source"]['created_at']!=None)):
          post_date = dateutil.parser.isoparse(post["_source"]['created_at'])
          if (post_date>after_date):
            final_list.append(post)
        else:
          print("Error no created_at for post", post)
    else:
      final_list = result_list

    print(len(final_list))

    def get_only_id_strs(n):
        return str(n["_id"])

    only_ids = list(map(get_only_id_strs, final_list))
    return only_ids

  def predict_for_collection(self, user_id, search_terms):
    post_ids = self.get_es_post_ids(search_terms)
    return self.predict_for_post_ids(str(user_id), post_ids)

  def predict_for_domain(self, domain_id, user_id):
    return self.predict_for_collection(user_id, { "domain_id": int(domain_id) })

  def predict_for_community(self, community_id, user_id):
    return self.predict_for_collection(user_id, { "community_id": int(community_id) })

  def predict_for_group(self, group_id, user_id):
    return self.predict_for_collection(user_id, { "group_id": int(group_id) })


