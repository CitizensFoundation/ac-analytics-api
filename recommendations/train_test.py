from lightfm_model_cache import LightFmModelCache
from elasticsearch import Elasticsearch, helpers, exceptions
import json

import datetime

from subprocess import check_output
from sklearn import metrics
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.evaluation import auc_score
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

    # cast generator as list to get length
    print('\nscan() scroll length:', len(list(resp)))

    # enumerate the documents
    for num, event in enumerate(resp):
        print('\n', num, '', event)
        raw_events.append(event)

    print(raw_events[0:10])

    print("RawEventsLength: ", len(raw_events))

    for event in raw_events:
        date = datetime.strptime(event.date, '%Y-%m-%dT%H:%M:%S.%fZ')
        timestamp = time.mktime(date.timetuple())

        pre_events.append({"timestamp": timestamp, "user_id": event.user_id, "post_id": event.post_id, "action": event.action })

    print(pre_events[0:10])

    lfm_post_ids = []
    lfm_timestamps = []
    lfm_user_ids = []
    lfm_actions = []
    for lfm_event in pre_events:
        lfm_post_ids.insert(0, lfm_event.post_id)
        lfm_timestamps.insert(0, lfm_event.timestamp)
        lfm_user_ids.insert(0, lfm_event.user_id)
        lfm_actions.insert(0, lfm_event.action)

    lfm_property_dict = {'post_id':lfm_post_ids, 'timestamp':lfm_timestamps, 'user_id': lfm_user_ids,
                        'action':lfm_actions}

    lfm_events = pd.DataFrame(lfm_property_dict, ["post_id","timestamp","user_id","action"])

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

    print(raw_posts[0:10])
    print("RawPostLength: ", len(raw_posts))

    for post in raw_posts:
        date = None
        if post.created_at:
            date = datetime.strptime(post.created_at, '%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            date = datetime.now();
        timestamp = time.mktime(date.timetuple())
        post_id = post.id

        property = "groupid"
        value = post.group_id

        pre_posts.append({property, value, post_id, timestamp})

        property = "communityid"
        value = post.community_id

        pre_posts.append({property, value, post_id, timestamp})

        property = "domain_id"
        value = post.domain_id

        pre_posts.append({property, value, post_id, timestamp})

        if post.category_id:
          property = "categoryid"
          value = post.category_id

          pre_posts.append({property, value, post_id, timestamp})

        property = "1"
        value = post.counter_endorsements_up > 0

        pre_posts.append({property, value, post_id, timestamp})

        property = "2"
        value = post.counter_endorsements_down > 0

        pre_posts.append({property, value, post_id, timestamp})

        property = "3"
        value = post.counter_points > 0

        pre_posts.append({property, value, post_id, timestamp})

        # TODO: Add, language, text hashes, the next five closes ideas and automatic keyword extraction features

    print(pre_posts[0:10])

    lfm_post_ids = []
    lfm_timestamps = []
    lfm_properties = []
    lfm_values = []
    for lfm_post in pre_posts:
        lfm_post_id.insert(0, lfm_post.post_id)
        lfm_timestamps.insert(0, lfm_post.timestamp)
        lfm_properties.insert(0, lfm_post.property)
        lfm_values.insert(0, lfm_post.value)

    print(lfm_post_ids[0:10])
    print(lfm_timestamps[0:10])
    print(lfm_properties[0:10])
    print(lfm_values[0:10])

    lfm_property_dict = {'post_id':lfm_post_ids, 'timestamp':lfm_timestamps, 'property':lfm_properties,
                        'value':lfm_values}

    lfm_posts = pd.DataFrame(lfm_property_dict, ["post_id","timestamp","property","value"])

    print(lfm_posts[0:10])
    print(len(lfm_posts))

    return lfm_posts

def get_trainingdata_from_es(cluster_id):
    posts = get_posts_from_es(cluster_id)
    events = get_data_from_es(cluster_id)
    category_tree = get_category_tree_from_es(cluster_id)

    return posts, events, category_tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#
# https://towardsdatascience.com/how-i-would-explain-building-lightfm-hybrid-recommenders-to-a-5-year-old-b6ee18571309
# https://www.kaggle.com/niyamatalmass/lightfm-hybrid-recommendation-system
# https://towardsdatascience.com/recommendation-system-in-python-lightfm-61c85010ce17
# https://www.kaggle.com/khacnghia97/recommend-ligthfm
# TODO: Automatically get keywords from posts texts and feed as tags into this algorithm
# TODO: Use groups
# TODO: Try to make single users of users with multiple accts, like Facebook login and SAML login

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

print("Loading")

cluster_id = 1

posts, events = get_trainingdata_from_es(cluster_id)

events = pd.read_csv('./events.csv')
category_tree = pd.read_csv('./category_tree.csv')
posts1 = pd.read_csv('./post_properties_part1.csv')
posts2 = pd.read_csv('./post_properties_part2.csv')
posts = pd.concat([posts1, posts2])

print("After load")

print(len(events))
print(len(posts))

print(type(events))


print(events[0:100])
print(posts[0:100])

#events = events[0:1000000]
#posts = posts[0:1000000]

user_activity_count = dict()
for row in events.itertuples():
    if row.user_id not in user_activity_count:
        user_activity_count[row.user_id] = {
            'new-post': 0, 'endorse': 0, 'oppose': 0,
            'new-point': 0, 'new-point-comment': 0, 'point-helpful': 0,
            'point-unhelpful': 0}
    if row.event == 'new-post':
        user_activity_count[row.user_id]['new-post'] += 1
    elif row.event == 'endorse':
        user_activity_count[row.user_id]['endorse'] += 1
    elif row.event == 'oppose':
        user_activity_count[row.user_id]['oppose'] += 1
    elif row.event == 'new-point':
        user_activity_count[row.user_id]['new-point'] += 1
    elif row.event == 'new-point-comment':
        user_activity_count[row.user_id]['new-point-comment'] += 1
    elif row.event == 'point-helpful':
        user_activity_count[row.user_id]['point-helpful'] += 1
    elif row.event == 'point-unhelpful':
        user_activity_count[row.user_id]['point-unhelpful'] += 1

d = pd.DataFrame(user_activity_count)
dataframe = d.transpose()

# Activity range
dataframe['new-post'] = dataframe['endorse'] + \
    dataframe['oppose'] + dataframe['new-point'] + \
    dataframe['new-point-comment'] + dataframe['point-helpful'] + \
    dataframe['point-unhelpful']

# removing users with only a single view
cleaned_data = dataframe[dataframe['activity'] != 1]
# all users contains the userids with more than 1 activity in the events (4lac)
all_users = set(cleaned_data.index.values)
all_posts = set(events['post_id'])
# todo: we need to clear posts which are only viewed once

#print(random.sample(all_users, 10))
#print(random.sample(all_posts, 10))

print("Before mapping")

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
user_to_post_matrix = sp.dok_matrix((n_users, n_posts), dtype=np.int8)
# We need to check whether we need to add the frequency of view, addtocart and transation.
# Currently we are only taking a single value for each row and column.

for row in events.itertuples():
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

#    current_value = user_to_post_matrix[mapped_user_id, mapped_post_id]
#    if value > current_value:
    user_to_post_matrix[mapped_user_id, mapped_post_id] = value

user_to_post_matrix = user_to_post_matrix.tocsr()

user_to_post_matrix.shape

filtered_posts = posts[posts.post_id.isin(all_posts)]

#print("After filtered posts")

# print(filtered_posts[0:100])
# print(user_to_post_matrix[0:100])

filtered_posts['post_id'] = filtered_posts['post_id'].apply(
    lambda x: post_id_to_index_mapping[x])

filtered_posts = filtered_posts.sort_values(
    'timestamp', ascending=False).drop_duplicates(['post_id', 'property'])
filtered_posts.sort_values(by='post_id', inplace=True)
post_to_property_matrix = filtered_posts.pivot(
    index='post_id', columns='property', values='value')

post_to_property_matrix.shape

useful_cols = list()
cols = post_to_property_matrix.columns
for col in cols:
    value = len(post_to_property_matrix[col].value_counts())
    if value < 50:
        useful_cols.insert(0, col)

post_to_property_matrix = post_to_property_matrix[useful_cols]

post_to_property_matrix_one_hot_sparse = pd.get_dummies(
    post_to_property_matrix)

print("Before lightfm")

post_to_property_matrix_one_hot_sparse.shape

post_to_property_matrix_sparse = csr_matrix(
    post_to_property_matrix_one_hot_sparse.values)

def make_train(ratings, pct_test=0.2):
    '''
    This function will take in the original user-post matrix and "mask" a percentage of the original ratings where a
    user-post interaction has taken place for use as a test set. The test set will contain all of the original ratings,
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix.

    parameters:

    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix.

    pct_test - The percentage of user-post interactions where an interaction took place that you want to mask in the
    training set for later comparison to the test set, which contains all of the original ratings.

    returns:

    training_set - The altered version of the original data with a certain percentage of the user-post pairs
    that originally had interaction set back to zero.

    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order
    compares with the actual interactions.

    user_inds - From the randomly selected user-post indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    test_set = ratings.copy()  # Make a copy of the original set to be the test set.
    # Store the test set as a binary preference matrix
    test_set[test_set != 0] = 1
    # Make a copy of the original data we can alter as our training set.
    training_set = ratings.copy()
    # Find the indices in the ratings data where an interaction exists
    nonzero_inds = training_set.nonzero()
    # Zip these pairs together of user,post index into list
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
    random.seed(0)  # Set the random seed to zero for reproducibility
    # Round the number of samples needed to the nearest integer
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs)))
    # Sample a random number of user-post pairs without replacement
    samples = random.sample(nonzero_pairs, num_samples)
    user_inds = [index[0] for index in samples]  # Get the user row indices
    post_inds = [index[1] for index in samples]  # Get the post column indices
    # Assign all of the randomly chosen user-post pairs to zero
    training_set[user_inds, post_inds] = 0
    # Get rid of zeros in sparse array storage after update to save space
    training_set.eliminate_zeros()
    # Output the unique list of user rows that were altered
    return training_set, test_set, list(set(user_inds))


print("Before make train =", datetime.now().strftime("%H:%M:%S"))


X_train, X_test, post_users_altered = make_train(
    user_to_post_matrix, pct_test=0.1)

print(X_train)
print(X_test)

print("Before fit partial =", datetime.now().strftime("%H:%M:%S"))

NUM_THREADS = 14

no_comp, lr, ep = 30, 0.01, 100
model = LightFM(no_components=no_comp, learning_rate=lr, loss='warp')
model.fit(
    X_train,
    post_features=post_to_property_matrix_sparse,
    epochs=ep,
    num_threads=NUM_THREADS,
    verbose=True)

print("After fit partial =", datetime.now().strftime("%H:%M:%S"))


test_auc = auc_score(model,
                     X_test,
                     post_features=post_to_property_matrix_sparse,
                     num_threads=NUM_THREADS).mean()
print('Hybrid test set AUC: %s' % test_auc)

LightFmModelCache.save_model(model, 99)
