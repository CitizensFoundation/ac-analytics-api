# Copyright (C) 2019, 2020 Íbúar ses / Citizens Foundation Iceland / Citizens Foundation America
# Author Robert Bjarnason
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os

import json
import sys
import json

from threading import Timer
import os
import os.path
from os import path

from flask import jsonify
from flask_restful import reqparse, Resource
from recommendations.predict import RecommendationPrediction
from recommendations.training_manager import RecTrainingManager
from recommendations.lightfm_model_cache import LightFmModelCache

from rq import Queue
from rq.job import Job
from worker import conn

from elasticsearch import Elasticsearch, NotFoundError

es_url = os.environ['AC_ANALYTICS_ES_URL'] if os.environ.get('AC_ANALYTICS_ES_URL')!=None else 'localhost:9200'

es = Elasticsearch(es_url)
queue = Queue(connection=conn, default_timeout=6000)

REC_TRAINING_TRIGGER_DEBOUNCE_TIME_SEC = 60*3

def deleteLockFileIfNeeded(object):
  if object["lockFilename"]!=None:
    if os.path.exists(object["lockFilename"]):
      os.remove(object["lockFilename"])

def start_recommendation_training(type, object):
    deleteLockFileIfNeeded(object)
    cluster_id = object["cluster_id"]
    print("start_recommendation_training", cluster_id, file=sys.stderr)
    training_manager = RecTrainingManager()
    model, user_id_map, user_features, item_id_map, item_features, interactions, user_feature_map = training_manager.train(cluster_id)
    LightFmModelCache.save(model, user_id_map, user_features, item_id_map, item_features, interactions, user_feature_map, cluster_id)

class AddPostAction(Resource):
    triggerTrainingTimer = {}

    def addToTriggerQueue(self, cluster_id, lockFilename):
        print("addToTriggerRecommendationsQueue", cluster_id)

        queue.enqueue_call(
            func=start_recommendation_training, args=("rec_training", {
                "cluster_id": cluster_id,
                "lockFilename": lockFilename
                }), result_ttl=1*60*60*1000, timeout=6000)

        AddPostAction.triggerTrainingTimer[cluster_id]=None;

    def post(self, cluster_id):
        parser = reqparse.RequestParser()
        parser.add_argument('action')
        parser.add_argument('esId')
        parser.add_argument('userId')
        parser.add_argument('postId')
        parser.add_argument('date')
        parser.add_argument('user_agent')
        parser.add_argument('ip_address')
        data = parser.parse_args()
        print(data)
        es.update(index='post_actions_'+cluster_id,id=data['esId'],body={'doc':data,'doc_as_upsert':True})

        if AddPostAction.triggerTrainingTimer.get(cluster_id)==None:
            lockFilename = "/tmp/acaRqInQueueRecommendations_{}".format(cluster_id);

            #TODO: Sometimes there are more than two trainings started at the same time so some subtle bug here
            #TODO: Check also for date, if to old delete
            if path.exists(lockFilename):
                print("Already in queue: "+lockFilename, file=sys.stderr)
            else:
                f = open(lockFilename, "w")
                f.write("x")
                f.close()

                print("Added rec training trigger timer", REC_TRAINING_TRIGGER_DEBOUNCE_TIME_SEC, file=sys.stderr)
                AddPostAction.triggerTrainingTimer[cluster_id] = Timer(REC_TRAINING_TRIGGER_DEBOUNCE_TIME_SEC,  self.addToTriggerQueue, [cluster_id, lockFilename])
                AddPostAction.triggerTrainingTimer[cluster_id].start()

        return jsonify({"ok":"true"})

class AddManyPostActions(Resource):
    def post(self, cluster_id):
        parser = reqparse.RequestParser()
        parser.add_argument('posts', action='append')
        post_data = parser.parse_args()
        for one_post in post_data['posts']:
            dict_post = eval(one_post)
            print(dict_post['esId'])
            es.update(index='post_actions_'+cluster_id,id=dict_post['esId'],body={'doc':dict_post,'doc_as_upsert':True})
        return jsonify({"ok":"true"})

class GetDomainRecommendations(Resource):
    def put(self, cluster_id, domain_id, user_id):
        parser = reqparse.RequestParser()
        parser.add_argument('user_agent')
        parser.add_argument('ip_address')
        parser.add_argument('dateOptions')
        user_data = parser.parse_args()
        prediction = RecommendationPrediction(cluster_id, user_data)
        return jsonify(prediction.predict_for_domain(domain_id,user_id))

class GetCommunityRecommendations(Resource):
    def put(self, cluster_id, community_id, user_id):
        parser = reqparse.RequestParser()
        parser.add_argument('user_agent')
        parser.add_argument('ip_address')
        parser.add_argument('date_options')
        user_data = parser.parse_args()
        prediction = RecommendationPrediction(cluster_id, user_data)
        return jsonify(prediction.predict_for_community(community_id,user_id))

class GetGroupRecommendations(Resource):
    def put(self, cluster_id, group_id, user_id):
        parser = reqparse.RequestParser()
        parser.add_argument('user_agent')
        parser.add_argument('ip_address')
        parser.add_argument('date_options')
        user_data = parser.parse_args()
        prediction = RecommendationPrediction(cluster_id, user_data)
        return jsonify(prediction.predict_for_group(group_id,user_id))