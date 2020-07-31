# Copyright (C) 2019,2020 Íbúar ses / Citizens Foundation Iceland / Citizens Foundation America
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

from flask import jsonify
from flask_restful import reqparse, Resource

from elasticsearch import Elasticsearch, NotFoundError
from world_cloud.worldcloud_manager import WorldCloudManager
if os.environ.get('AC_ANALYTICS_API_URL'):
    api_url = os.environ['AC_ANALYTICS_API_URL']
else:
    api_url = '/api/v1'

es_url = os.environ['AC_ANALYTICS_ES_URL'] if os.environ.get('AC_ANALYTICS_ES_URL')!=None else 'localhost:9200'

master_api_key = os.environ['AC_ANALYTICS_MASTER_API_KEY']

es = Elasticsearch(es_url)

class GetPostWordCloud(Resource):
    def get(self, cluster_id, post_id):
        wordcloud = WorldCloudManager("points_"+cluster_id,"point",{"post_id": post_id, "cluster_id": cluster_id})
        return jsonify(wordcloud.getWordCloud())

class GetGroupWordCloud(Resource):
    def get(self, cluster_id, group_id):
        wordcloud = WorldCloudManager("posts_"+cluster_id,"post",{"group_id": group_id, "cluster_id": cluster_id})
        return jsonify(wordcloud.getWordCloud())

class GetCommunityWordCloud(Resource):
    def get(self, cluster_id, community_id):
        wordcloud = WorldCloudManager("posts_"+cluster_id,"post",{"community_id": community_id, "cluster_id": cluster_id})
        return jsonify(wordcloud.getWordCloud())

class GetDomainWordCloud(Resource):
    def get(self, cluster_id, domain_id):
        wordcloud = WorldCloudManager("posts_"+cluster_id,"post",{"domain_id": domain_id, "cluster_id": cluster_id})
        return jsonify(wordcloud.getWordCloud())

