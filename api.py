# Copyright (C) 2019 Íbúar ses / Citizens Foundation Iceland / Citizens Foundation America
# Authors Atli Jasonarson & Robert Bjarnason
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

from threading import Timer

from flask import Flask, jsonify
from flask_restful import request, reqparse, Resource, Api
from flask_cors import CORS

from rq import Queue
from rq.job import Job

from elasticsearch import Elasticsearch

from training.training import triggerPostTraining, triggerPointTraining, triggerArticleTraining
from worker import conn
from lemmatizer.lemmatizer import getLemmatizedText
from similarities.similarities import PostSimilarity
from training.weights_manager import WeightsManager

from controllers.similarities import PostList, PointList, DomainList, CommunityList, GroupList, FindSimilarPosts
from controllers.similarities import GetCommunityPostsWithWeights, GetGroupPostsWithWeights, GetDomainPostsWithWeights, GetPostPointsWithWeights
from controllers.similarities import TriggerDomainPostTraining, TriggerCommunityPostTraining, TriggerGroupPostTraining, TriggerPostPointsTraining

from controllers.wordcloud import GetCommunityWordCloud, GetDomainWordCloud, GetGroupWordCloud, GetPostWordCloud

from controllers.conversion import ConvertDocxSurveyToJSON

from controllers.recommendations import AddPostAction
from controllers.recommendations import AddManyPostActions
from controllers.recommendations import GetDomainRecommendations
from controllers.recommendations import GetCommunityRecommendations
from controllers.recommendations import GetGroupRecommendations

if os.environ.get('AC_ANALYTICS_API_URL'):
    api_url = os.environ['AC_ANALYTICS_API_URL']
else:
    api_url = '/api/v1'

es_url = os.environ['AC_ANALYTICS_ES_URL'] if os.environ.get('AC_ANALYTICS_ES_URL')!=None else 'localhost:9200'

master_api_key = os.environ['AC_ANALYTICS_MASTER_API_KEY']

app = Flask(__name__)
CORS(app)
api = Api(app)

es = Elasticsearch(es_url)
queue = Queue(connection=conn, default_timeout=6000)

@app.before_request
def before_request():
    headers = request.headers
    auth = headers.get("X-API-KEY")
    if (auth!=master_api_key):
        return jsonify({"message": "ERROR: You are not authorized"}), 401

if (len(sys.argv)>1):
    if (sys.argv[1]=="deleteAllIndexesESger32jh8"):
        if es.indices.exists("posts"):
           es.indices.delete("posts")
        if es.indices.exists("points"):
           es.indices.delete("points")
        if es.indices.exists("articles"):
           es.indices.delete("articles")
        if es.indices.exists("domains"):
           es.indices.delete("domains")
        if es.indices.exists("communities"):
           es.indices.delete("communities")
        if es.indices.exists("groups"):
           es.indices.delete("groups")
        if es.indices.exists("policyGames"):
           es.indices.delete("policyGames")
        if es.indices.exists("similarityweights"):
           es.indices.delete("similarityweights")
        print("HAVE DELETED ALL ES INDICES")


class Healthcheck(Resource):
    def get(self):
       # TODO: Do some checking
       print("Healthcheck")

# Similarities APIs
api.add_resource(PostList, api_url+'/posts/<cluster_id>/<post_id>')
api.add_resource(PointList, api_url+'/points/<cluster_id>/<point_id>')
api.add_resource(DomainList, api_url+'/domains/<cluster_id>/<domain_id>')
api.add_resource(CommunityList, api_url+'/communities/<cluster_id>/<community_id>')
api.add_resource(GroupList, api_url+'/groups/<cluster_id>/<group_id>')

api.add_resource(FindSimilarPosts, api_url+'/find_similar_posts/<cluster_id>')

api.add_resource(GetCommunityPostsWithWeights, api_url+'/similarities_weights/community/<cluster_id>/<community_id>')
api.add_resource(GetGroupPostsWithWeights, api_url+'/similarities_weights/group/<cluster_id>/<group_id>')
api.add_resource(GetDomainPostsWithWeights, api_url+'/similarities_weights/domain/<cluster_id>/<domain_id>')
api.add_resource(GetPostPointsWithWeights, api_url+'/similarities_weights/post/<cluster_id>/<post_id>')

api.add_resource(GetPostWordCloud, api_url+'/wordclouds/post/<cluster_id>/<post_id>')
api.add_resource(GetGroupWordCloud, api_url+'/wordclouds/group/<cluster_id>/<group_id>')
api.add_resource(GetCommunityWordCloud, api_url+'/wordclouds/community/<cluster_id>/<community_id>')
api.add_resource(GetDomainWordCloud, api_url+'/wordclouds/domain/<cluster_id>/<domain_id>')

api.add_resource(TriggerDomainPostTraining, api_url+'/trigger_similarities_training/domain/<cluster_id>/<domain_id>')
api.add_resource(TriggerCommunityPostTraining, api_url+'/trigger_similarities_training/community/<cluster_id>/<community_id>')
api.add_resource(TriggerGroupPostTraining, api_url+'/trigger_similarities_training/group/<cluster_id>/<group_id>')

api.add_resource(TriggerPostPointsTraining, api_url+'/trigger_similarities_training/post/<cluster_id>/<post_id>')

api.add_resource(ConvertDocxSurveyToJSON, api_url+'/convert_doc_x_survey_to_json')

api.add_resource(AddPostAction, api_url+'/addPostAction/<cluster_id>')

api.add_resource(AddManyPostActions, api_url+'/addManyPostActions/<cluster_id>')

api.add_resource(GetDomainRecommendations, api_url+'/recommendations/domain/<cluster_id>/<domain_id>/<user_id>')
api.add_resource(GetCommunityRecommendations, api_url+'/recommendations/community/<cluster_id>/<community_id>/<user_id>')
api.add_resource(GetGroupRecommendations, api_url+'/recommendations/group/<cluster_id>/<group_id>/<user_id>')

api.add_resource(Healthcheck, api_url+'/healthcheck')

# Anonymized export APIs

if __name__ == "__main__":
    app.run(host='0.0.0.0')
