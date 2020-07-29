from lightfm_model_cache import LightFmModelCache
from elasticsearch import Elasticsearch, helpers, exceptions
import json

import datetime
import time

from subprocess import check_output
from sklearn import metrics
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split

from datetime import datetime
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

sys.path.append(".")

def get_events_from_es(cluster_id):
    raw_events = []
    pre_events = []
    es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get(
        'AC_SIM_ES_URL') != None else 'localhost:9200'
    es_client = Elasticsearch(es_url)
    resp = helpers.scan(
        es_client,
        index='post_actions_'+str(cluster_id),
        scroll='3m',
        size=1000
    )

    # returns a generator object
    print(type(resp))

    events_list = list(resp)

    print(len(events_list))

    # enumerate the documents
    for event in events_list:
        raw_events.append(event)

    print(raw_events[0:10])

    print("RawEventsLength: ", len(raw_events))

    lfm_post_ids = []
    lfm_timestamps = []
    lfm_user_ids = []
    lfm_actions = []
    for event in raw_events:
        date = datetime.strptime(event['_source']["date"], '%Y-%m-%dT%H:%M:%S.%fZ')
        timestamp = int("{:%s}".format(date))
        lfm_post_ids.append(str(event['_source']["postId"]))
        lfm_timestamps.append(timestamp)
        lfm_user_ids.append(str(event['_source']["userId"]))
        lfm_actions.append(event['_source']["action"])

    lfm_property_dict = {'post_id':lfm_post_ids, 'timestamp':lfm_timestamps, 'user_id': lfm_user_ids,
                        'action':lfm_actions}

    lfm_events = pd.DataFrame(lfm_property_dict)

    print(lfm_events[0:10])
    print(len(lfm_events))

    return lfm_events

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

    # returns a generator object
    print(type(resp))

    # cast generator as list to get length
    #print('\nscan() scroll length:', len(list(resp)))

    item_list = list(resp)

    print(len(item_list))

    # enumerate the documents
    for item in item_list:
        raw_posts.append(item)

    #print(raw_posts[0:10])
    print("RawPostLength: ", len(raw_posts))

    for post in raw_posts:
        date = None
        timestamp = None
        if 'created_at' in post['_source']:
            date = datetime.strptime( post['_source']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
            timestamp = int("{:%s}".format(date))
        else:
            date = datetime.now();
            timestamp = int("{:%s}".format(date))

        post_id = post['_id']

        property = "groupid"
        value = post['_source']['group_id']

        pre_posts.append({"property": property, "value": value, "post_id": str(post_id), "timestamp": timestamp})

        property = "communityid"
        value = post['_source']['community_id']

        #pre_posts.append({"property": property, "value": value, "post_id": post_id, "timestamp": timestamp})

        property = "domain_id"
        value = post['_source']['domain_id']

        #pre_posts.append({"property": property, "value": value, "post_id": post_id, "timestamp": timestamp})

        if 'category_id' in post['_source']:
          property = "categoryid"
          value = post['_source']['category_id']
          #pre_posts.append({"property": property, "value": value, "post_id": post_id, "timestamp": timestamp})

        property = "1"
        value = post['_source']['counter_endorsements_up'] > 0

        #pre_posts.append({"property": property, "value": value, "post_id": post_id, "timestamp": timestamp})

        property = "2"
        value = post['_source']['counter_endorsements_down'] > 0

        #pre_posts.append({"property": property, "value": value, "post_id": post_id, "timestamp": timestamp})

        property = "3"
        value = post['_source']['counter_points'] > 0

        #pre_posts.append({"property": property, "value": value, "post_id": post_id, "timestamp": timestamp})

        # TODO: Add, language, text hashes, the next five closes ideas and automatic keyword extraction features

    print("pre_posts len: ", len(raw_posts))
    print(pre_posts[0:10])

    lfm_post_ids = []
    lfm_timestamps = []
    lfm_properties = []
    lfm_values = []
    for lfm_post in pre_posts:
        lfm_post_ids.append(int(lfm_post["post_id"]))
        lfm_timestamps.append(lfm_post["timestamp"])
        lfm_properties.append(lfm_post["property"])
        lfm_values.append(lfm_post["value"])

    print("Post ids")
    print(lfm_post_ids[0:10])
    print("Timestamps")
    print(lfm_timestamps[0:10])
    print("Properties")
    print(lfm_properties[0:10])
    print("Values")
    print(lfm_values[0:10])

    lfm_property_dict = {'post_id':lfm_post_ids, 'timestamp':lfm_timestamps, 'property':lfm_properties,
                        'value':lfm_values}

    lfm_posts = pd.DataFrame(lfm_property_dict)

    print(lfm_posts[0:10])
    print(len(lfm_posts))

    return lfm_posts

def get_trainingdata_from_es(cluster_id):
    posts = get_posts_from_es(cluster_id)
    events = get_events_from_es(cluster_id)
#    category_tree = get_category_tree_from_es(cluster_id)

    return posts, events

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#
# https://towardsdatascience.com/how-i-would-explain-building-lightfm-hybrid-recommenders-to-a-5-year-old-b6ee18571309
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

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

print("Loading")

cluster_id = 1

posts, events = get_trainingdata_from_es(cluster_id)

#events = pd.read_csv('./events.csv')
#category_tree = pd.read_csv('./category_tree.csv')
#posts1 = pd.read_csv('./post_properties_part1.csv')
#posts2 = pd.read_csv('./post_properties_part2.csv')
#posts = pd.concat([posts1, posts2])

print("After load")

print(len(events))
print(len(posts))

print(type(events))

print(events[0:100])
print(posts[0:100])

#events = events[0:1000000]
#posts = posts[0:1000000]

all_users = set(events['user_id'])
all_posts = set(events['post_id'])

print("Before mapping to consequtives integers")

user_id_to_index_mapping = {}
post_id_to_index_mapping = {}
vid = 0
iid = 0

for row in events.itertuples():
    if row.user_id in all_users and row.user_id not in user_id_to_index_mapping:
        user_id_to_index_mapping[row.user_id] = vid
        vid = vid + 1

    if row.post_id in all_posts and row.post_id not in post_id_to_index_mapping:
        post_id_to_index_mapping[row.post_id] = iid
        iid = iid + 1

n_users = len(all_users)
n_posts = len(all_posts)
print("NPOST", n_posts)
print("NUSERS", n_users)
user_to_post_matrix = sp.dok_matrix((n_users, n_posts), dtype=np.int8)

# We need to check whether we need to add the frequency of view, addtocart and transation.
# Currently we are only taking a single value for each row and column.

for row in events.itertuples():
    #print(row)
    if row.user_id not in all_users:
        continue

    mapped_user_id = user_id_to_index_mapping[row.user_id]
    mapped_post_id = post_id_to_index_mapping[row.post_id]

    value = 0
    if row.action == 'new-post':
        value = 4
    elif row.action == 'endorse':
        value = 2
    elif row.action == 'oppose':
        value = 1
    elif row.action == 'new-point':
        value = 3
    elif row.action == 'new-point-comment':
        value = 3
    elif row.action == 'point-helpful':
        value = 2
    elif row.action == 'point-unhelpful':
        value = 1

    current_value = user_to_post_matrix[mapped_user_id, mapped_post_id]
    if current_value and current_value>0:
      user_to_post_matrix[mapped_user_id, mapped_post_id] = current_value+value
    else:
      user_to_post_matrix[mapped_user_id, mapped_post_id] = value

#user_to_post_matrix = user_to_post_matrix.tocsr()

#user_to_post_matrix = user_to_post_matrix.tocoo()

#user_to_post_matrix.shape

print(user_to_post_matrix)

#filtered_posts = posts[posts.post_id.isin(all_posts)]

#print(filtered_posts)

#print(len(filtered_posts))

#print("After filtered posts")

# print(filtered_posts[0:100])
# print(user_to_post_matrix[0:100])

#filtered_posts['post_id'] = filtered_posts['post_id'].apply(
#    lambda x: post_id_to_index_mapping[x])

#filtered_posts = filtered_posts.sort_values(
#    'timestamp', ascending=False).drop_duplicates(['post_id', 'property'])

#filtered_posts.sort_values(by='post_id', inplace=True)

#post_to_property_matrix = filtered_posts.pivot(
#    index='post_id', columns='property', values='value')

#post_to_property_matrix.shape

#useful_cols = list()
##cols = post_to_property_matrix.columns
#for col in cols:
#    value = len(post_to_property_matrix[col].value_counts())
#    if value < 50:
#        useful_cols.insert(0, col)

#post_to_property_matrix = post_to_property_matrix[useful_cols]

#post_to_property_matrix_one_hot_sparse = pd.get_dummies(
#    post_to_property_matrix)

print("Before lightfm")

#post_to_property_matrix_one_hot_sparse.shape

#post_to_property_matrix_sparse = csr_matrix(
#    post_to_property_matrix_one_hot_sparse.values)

print("Before make train =", datetime.now().strftime("%H:%M:%S"))

X_train, X_test = random_train_test_split(user_to_post_matrix, random_state=np.random.RandomState(3))

#print(X_train)
#print(X_test)

print("Before fit partial =", datetime.now().strftime("%H:%M:%S"))

NUM_THREADS = 14
NUM_THREADS = 2
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

no_comp, lr, ep = 30, 0.01, 20
model = LightFM(no_components=NUM_COMPONENTS, item_alpha=ITEM_ALPHA, loss='warp')
model.fit(
    X_train,
#    item_features=post_to_property_matrix_sparse,
    epochs=NUM_EPOCHS,
    num_threads=NUM_THREADS,
    verbose=True)

print("After fit partial =", datetime.now().strftime("%H:%M:%S"))

test_auc = auc_score(model,
                     X_train,
                     num_threads=NUM_THREADS).mean()
print('Train set AUC: %s' % test_auc)

test_auc = auc_score(model,
                     X_test,
                     train_interactions=X_test,
                     num_threads=NUM_THREADS).mean()
print('Test set AUC: %s' % test_auc)

LightFmModelCache.save_model(model, 99)
