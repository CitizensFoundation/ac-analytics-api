NUM_THREADS = 8
NUM_COMPONENTS = 30
NUM_EPOCHS = 42
ITEM_ALPHA = 1e-6

# Originally based on this https://towardsdatascience.com/how-i-would-explain-building-lightfm-hybrid-recommenders-to-a-5-year-old-b6ee18571309
# https://www.kaggle.com/niyamatalmass/lightfm-hybrid-recommendation-system
# https://towardsdatascience.com/recommendation-system-in-python-lightfm-61c85010ce17
# https://www.kaggle.com/khacnghia97/recommend-ligthfm
# TODO: Automatically get keywords from posts texts and feed as tags into this algorithm
# TODO: Use groups
# TODO: https://github.com/aolieman/wayward check for keyword extraction
# TODO: For user features, user agent https://pypi.org/project/user-agents/
# TODO: For user geo features https://pypi.org/project/ip2geotools/
# TODO: Research/make use of this: https://engineering.linkedin.com/blog/2020/open-sourcing-detext
# TODO: Try to make single users of users with multiple accts, like Facebook login and SAML login

from recommendations.lightfm_model_cache import LightFmModelCache
from elasticsearch import Elasticsearch, helpers, exceptions
import json

from datetime import datetime
import time

from subprocess import check_output
from sklearn import metrics
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset

from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as sp
import random
from scipy.sparse import vstack
from scipy import sparse
from scipy.sparse.linalg import spsolve
import os
import sys
import pickle
from user_agents import parse

sys.path.append(".")

def get_posts_from_es(cluster_id):
    raw_posts = []
    pre_posts = []
    es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get(
        'AC_SIM_ES_URL') != None else 'localhost:9200'
    es_client = Elasticsearch(es_url)
    resp = helpers.scan(
        es_client,
        query={'query': {'match_all': {}}},
        index='posts_'+str(cluster_id),
        scroll='3m'
    )

    return list(resp)

def get_events_from_es(cluster_id):
    es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get(
    'AC_SIM_ES_URL') != None else 'localhost:9200'
    es_client = Elasticsearch(es_url)
    resp = helpers.scan(
        es_client,
        index='post_actions_'+str(cluster_id),
        scroll='3m',
        size=2500
    )

    return list(resp)

def build_item_features_dataframe(cluster_id, interactions):
    print("build_item_features_dataframe")

    raw_posts = get_posts_from_es(cluster_id)

    print("RawPostLength: ", len(raw_posts))

    post_ids = []

    group_id = []
    community_id = []
    domain_id = []

    category_id = []

    has_up_votes = []
    has_down_votes = []
    got_points = []

    item_tuple = []

    for post in raw_posts:
        post_id = str(post['_id'])

        if post_id in interactions:
            post_ids.append(post_id)

            group_id.append(post['_source']['group_id'])
            community_id.append(post['_source']['community_id'])
            domain_id.append(post['_source']['domain_id'])

            if 'category_id' in post['_source']:
                category_id.append(post['_source']['category_id'])
            else:
                category_id.append(0)

            has_up_votes.append(True if post['_source']['counter_endorsements_up'] > 0 else False)
            has_down_votes.append(True if post['_source']['counter_endorsements_down'] > 0 else False)
            got_points.append(True if post['_source']['counter_points'] > 0 else False)

            features_array = []
            features_array.append("group_id:"+str(group_id[-1]))
            features_array.append("community_id:"+str(community_id[-1]))
            features_array.append("domain_id:"+str(domain_id[-1]))
            features_array.append("category_id:"+str(category_id[-1]))

            features_array.append("has_up_votes:"+str(has_up_votes[-1]))
            features_array.append("has_down_votes:"+str(has_down_votes[-1]))

            features_array.append("got_points:"+str(got_points[-1]))

            item_tuple.append((post_id, features_array))

    dataframe_items_features = pd.DataFrame({
        'item': post_ids,

        'group_id': group_id,
        'community_id': community_id,
        'domain_id': domain_id,
        'category_id': category_id,

        'has_up_votes': has_up_votes,
        'has_down_votes': has_down_votes,

        'got_points': got_points
    },  columns = ['item', 'group_id', 'community_id', 'domain_id', 'category_id', 'has_up_votes', 'has_down_votes','got_points'])

    return dataframe_items_features, item_tuple

def get_item_features_tuple(items_features):
    item_tuple = []

    for post_id in items_features:
        features_array = []
        features_array.append("group_id:"+items_features[post_id]["group_id"])
        features_array.append("community_id:"+items_features[post_id]["community_id"])
        features_array.append("domain_id:"+items_features[post_id]["domain_id"])
        features_array.append("category_id:"+items_features[post_id]["category_id"])

        features_array.append("has_up_votes:"+items_features[post_id]["has_up_votes"])
        features_array.append("has_down_votes:"+items_features[post_id]["has_down_votes"])

        features_array.append("got_points:"+items_features[post_id]["got_points"])

        item_tuple.append(item_id, features_array)

def setup_user_features(user_features_dict, event):
    is_mobile = 2
    is_tablet = 2
    is_pc = 2
    is_bot = 2

    browser_family = "0"

    os_family = "0"

    device_family =  "0"
    device_brand =  "0"

    if 'user_agent' in event["_source"]:
        user_agent = parse(event["_source"]['user_agent'])
        is_mobile = user_agent.is_mobile
        is_tablet = user_agent.is_tablet
        is_pc = user_agent.is_pc
        is_bot = user_agent.is_bot

        os_family = user_agent.os.family

        browser_family = user_agent.browser.family

        device_family = user_agent.device.family
        device_brand = user_agent.device.brand

    user_features_dict["is_mobile"] = is_mobile
    user_features_dict["is_tablet"] = is_tablet
    user_features_dict["is_pc"] = is_pc
    user_features_dict["is_bot"] = is_bot

    user_features_dict["browser_family"] = browser_family

    user_features_dict["os_family"] = os_family

    user_features_dict["device_family"] = device_family
    user_features_dict["device_brand"] = device_brand if device_brand else "0"

# TODO: Cache those in files
def get_user_features_tuple(users_features):
    user_tuple = []

    for user_id in users_features:
        features_array = []
        features_array.append("is_mobile:"+str(users_features[user_id]["is_mobile"]))
        features_array.append("is_tablet:"+str(users_features[user_id]["is_tablet"]))
        features_array.append("is_pc:"+str(users_features[user_id]["is_pc"]))
        features_array.append("is_bot:"+str(users_features[user_id]["is_bot"]))

        features_array.append("browser_family:"+users_features[user_id]["browser_family"])

        features_array.append("os_family:"+users_features[user_id]["os_family"])

        features_array.append("device_family:"+users_features[user_id]["device_family"])
        features_array.append("device_brand:"+users_features[user_id]["device_brand"])

        user_tuple.append((user_id, features_array))

    return user_tuple


def get_users_features_dataframe(users_features):
    user_ids_for_features = []

    user_feature_is_mobile = []
    user_feature_is_tablet = []
    user_feature_is_pc = []
    user_feature_is_bot = []

    user_feature_browser_family = []

    user_feature_os_family = []

    user_feature_device_family = []
    user_feature_device_brand = []

    for user_id in users_features:
        user_ids_for_features.append(user_id)

        user_feature_is_mobile.append(users_features[user_id]["is_mobile"])
        user_feature_is_tablet.append(users_features[user_id]["is_tablet"])
        user_feature_is_pc.append(users_features[user_id]["is_pc"])
        user_feature_is_bot.append(users_features[user_id]["is_bot"])

        user_feature_browser_family.append(users_features[user_id]["browser_family"])

        user_feature_os_family.append(users_features[user_id]["os_family"])

        user_feature_device_family.append(users_features[user_id]["device_family"])
        user_feature_device_brand.append(users_features[user_id]["device_brand"])

    dataframe_users_features = pd.DataFrame({
        'user': user_ids_for_features,

        'is_mobile': user_feature_is_mobile,
        'is_tablet': user_feature_is_tablet,
        'is_pc': user_feature_is_pc,
        'is_bot': user_feature_is_bot,

        'browser_family': user_feature_browser_family,

        'os_family': user_feature_os_family,

        'device_family': user_feature_device_family,
        'device_brand': user_feature_device_brand
    },  columns = ['user', 'is_mobile', 'is_tablet', 'is_pc', 'is_bot', 'browser_family',
                'os_family', 'device_family', 'device_brand'])

    return dataframe_users_features

def format_items_features(features):
    uf = []
    col = ['group_id']*len(features['group_id'].unique()) + \
        ['community_id']*len(features['community_id'].unique()) + \
        ['domain_id']*len(features['domain_id'].unique()) + \
        ['category_id']*len(features['category_id'].unique()) + \
        ['has_up_votes']*len(features['has_up_votes'].unique()) + \
        ['has_down_votes']*len(features['has_down_votes'].unique()) + \
        ['got_points']*len(features['got_points'].unique())
    unique_f1 = list(features["group_id"].unique()) + \
        list(features["community_id"].unique()) + \
        list(features["domain_id"].unique()) + \
        list(features["category_id"].unique()) + \
        list(features["has_up_votes"].unique()) + \
        list(features["has_down_votes"].unique()) + \
        list(features["got_points"].unique())
    #print('f1:', unique_f1)
    for x,y in zip(col, unique_f1):
        res = str(x)+ ":" +str(y)
        uf.append(res)

    return uf

def format_users_features(features):
    uf = []
    col = ['is_mobile']*len(features['is_mobile'].unique()) + \
        ['is_tablet']*len(features['is_tablet'].unique()) + \
        ['is_pc']*len(features['is_pc'].unique()) + \
        ['is_bot']*len(features['is_bot'].unique()) + \
        ['browser_family']*len(features['browser_family'].unique()) + \
        ['os_family']*len(features['os_family'].unique()) + \
        ['device_family']*len(features['device_family'].unique()) + \
        ['device_brand']*len(features['device_brand'].unique())
    unique_f1 = list(features["is_mobile"].unique()) + \
        list(features["is_tablet"].unique()) + \
        list(features["is_pc"].unique()) + \
        list(features["is_bot"].unique()) + \
        list(features["browser_family"].unique()) + \
        list(features["os_family"].unique()) + \
        list(features["device_family"].unique()) + \
        list(features["device_brand"].unique())
    #print('f1:', unique_f1)
    for x,y in zip(col, unique_f1):
        res = str(x)+ ":" +str(y)
        uf.append(res)

    return uf

def create_interactions_and_features(events_list, cluster_id):
    print("create_interactions_and_features", file=sys.stderr)
    interactions = {}
    users_features = {}
    posts_features = {}

    for event in events_list:
        post_id = str(event['_source']["postId"])
        user_id = str(event['_source']["userId"])

        if post_id not in interactions:
            interactions[post_id] = {}
        if user_id not in interactions[post_id]:
            interactions[post_id][user_id] = 0

        action = event['_source']["action"]

        rating = 0
        if action == 'new-post':
            rating = 4
        elif action == 'endorse':
            rating = 2
        elif action == 'oppose':
            rating = 1
        elif action == 'new-point':
            rating = 3
        elif action == 'new-point-comment':
            rating = 3
        elif action == 'point-helpful':
            rating = 2
        elif action == 'point-unhelpful':
            rating = 1
        else:
            print("Error can't find action named "+action)

        if rating>0:
            interactions[post_id][user_id] += rating

        if user_id not in users_features:
            users_features[user_id] = {}

        setup_user_features(users_features[user_id], event)

    post_ids = []
    user_ids = []
    ratings = []

    for post_id in interactions:
        for user_id in interactions[post_id]:
            post_ids.append(post_id)
            user_ids.append(user_id)
            ratings.append(interactions[post_id][user_id])

    dataframe_interactions = pd.DataFrame({
        'user': user_ids,
        'item': post_ids,
        'r': ratings
    },  columns = ['user', 'item', 'r'])


    dataframe_users_features = get_users_features_dataframe(users_features)

    user_tuple = get_user_features_tuple(users_features)

    dataframe_item_features, item_tuple = build_item_features_dataframe(cluster_id, interactions)

    return dataframe_interactions, dataframe_users_features, dataframe_item_features, user_tuple, item_tuple

def create_datasets(cluster_id):

    events_list = get_events_from_es(cluster_id)

    dataframe_interactions, dataframe_users_features, dataframe_item_features, user_tuple, item_tuple = create_interactions_and_features(events_list, cluster_id)

    print(dataframe_interactions, cluster_id, file=sys.stderr)
    print(dataframe_users_features, cluster_id, file=sys.stderr)
    print(dataframe_item_features, cluster_id, file=sys.stderr)

    #print(user_tuple)
   # print(item_tuple)

    user_features = format_users_features(dataframe_users_features)

    #print(user_features)

    item_features = format_items_features(dataframe_item_features)

    #print(item_features)

    dataset = Dataset()

    dataset.fit(
            dataframe_interactions['user'].unique(), # all the users
            dataframe_interactions['item'].unique(), # all the items
            user_features = user_features,
            item_features = item_features
    )

    (interactions, weights) = dataset.build_interactions([(x[0], x[1], x[2]) for x in dataframe_interactions.values ])

#    print(interactions)
#    print(weights)

    final_user_features = dataset.build_user_features(user_tuple, normalize= False)

    final_item_features = dataset.build_item_features(item_tuple, normalize= False)

    return dataset, interactions, weights, final_item_features, final_user_features

###############################################

class RecTrainingManager:
    def train(self, cluster_id):
        print("Train Start Time =", datetime.now().strftime("%H:%M:%S"), cluster_id, file=sys.stderr)

        #TODO: Optimize dataset creation by using some sort of caching and partial updates
        dataset, interactions, weights, item_features, user_features = create_datasets(cluster_id)
        user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

        print("Before fit =", datetime.now().strftime("%H:%M:%S"), cluster_id, file=sys.stderr)

        no_comp, lr, ep = 30, 0.01, 20
        model = LightFM(no_components=NUM_COMPONENTS, item_alpha=ITEM_ALPHA, loss='warp')
        model.fit(
            interactions,
            item_features=item_features,
            user_features=user_features,
            sample_weight= weights,
            epochs=NUM_EPOCHS,
            num_threads=NUM_THREADS,
            verbose=True)

        print("After fit =", datetime.now().strftime("%H:%M:%S"), cluster_id, file=sys.stderr)

        return model, user_id_map, user_features, item_id_map, item_features, interactions, user_feature_map

    def train_with_test_data(self, cluster_id):
        print("Start Time =", datetime.now().strftime("%H:%M:%S"))

        dataset, interactions, weights, item_features, user_features = create_datasets(cluster_id)
        user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

        train_data, test_data = random_train_test_split(interactions, random_state=np.random.RandomState(3))
        train_weights, test_weights = random_train_test_split(weights, random_state=np.random.RandomState(3))

        print("Before fit =", datetime.now().strftime("%H:%M:%S"))

        no_comp, lr, ep = 30, 0.01, 20
        model = LightFM(no_components=NUM_COMPONENTS, item_alpha=ITEM_ALPHA, loss='warp')
        model.fit(
            train_data,
            item_features=item_features,
            user_features=user_features,
            sample_weight= train_weights,
            epochs=NUM_EPOCHS,
            num_threads=NUM_THREADS,
            verbose=True)

        print("After fit =", datetime.now().strftime("%H:%M:%S"))

        return model, user_id_map, user_features, item_id_map, item_features, train_data, test_data
