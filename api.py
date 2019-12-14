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
from simiarities.similarities import PostSimilarity
from training.weights_manager import WeightsManager

from controllers.similarities import PostList, PointList, DomainList, CommunityList, GroupList, FindSimilarPosts
from controllers.similarities import GetCommunityPostsWithWeights, GetGroupPostsWithWeights, GetDomainPostsWithWeights, GetPostPointsWithWeights

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
queue = Queue(connection=conn)

@app.before_request
def before_request():
    headers = request.headers
    auth = headers.get("X-Api-Key")
    if (auth!=master_api_key):
        return jsonify({"message": "ERROR: Unauthorized"}), 401

if (len(sys.argv)>1):
    if (sys.argv[1]=="deleteAllIndexesES"):
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

# Similarities APIs
api.add_resource(PostList, api_url+'/posts/<cluster_id>/<post_id>')
api.add_resource(PointList, api_url+'/points/<cluster_id>/<point_id>')
api.add_resource(DomainList, api_url+'/domains/<cluster_id>/<domain_id>')
api.add_resource(CommunityList, api_url+'/communities/<cluster_id>/<community_id>')
api.add_resource(GroupList, api_url+'/groups/<cluster_id>/<group_id>')
api.add_resource(FindSimilarPosts, api_url+'/find_similar_posts/<cluster_id>')
api.add_resource(GetCommunityPostsWithWeights, api_url+'/getCommunityPostsWithWeights/<cluster_id><community_id>')
api.add_resource(GetGroupPostsWithWeights, api_url+'/getGroupPostsWithWeights/<cluster_id>/<group_id>')
api.add_resource(GetDomainPostsWithWeights, api_url+'/getDomainPostsWithWeights/<cluster_id>/<domain_id>')
api.add_resource(GetPostPointsWithWeights, api_url+'/GetPostPointsWithWeights/<cluster_id>/<post_id>')

# Anonymized export APIs

if __name__ == "__main__":
    app.run(host='0.0.0.0')
