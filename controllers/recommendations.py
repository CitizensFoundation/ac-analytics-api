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

from flask import jsonify
from flask_restful import reqparse, Resource

from elasticsearch import Elasticsearch, NotFoundError

es_url = os.environ['AC_ANALYTICS_ES_URL'] if os.environ.get('AC_ANALYTICS_ES_URL')!=None else 'localhost:9200'

parser = reqparse.RequestParser()

es = Elasticsearch(es_url)

def get_recommendations

class AddPostAction(Resource):
    def post(self, cluster_id):
        parser.add_argument('action')
        parser.add_argument('esId')
        parser.add_argument('userId')
        parser.add_argument('postId')
        parser.add_argument('date')
        parser.add_argument('user_agent')
        parser.add_argument('ip_address')
        data = parser.parse_args()

        es.update(index='post_actions_'+cluster_id,id=data['esId'],body={'doc':data,'doc_as_upsert':True})
        return jsonify({"ok":"true"})

class AddManyPostActions(Resource):
    def post(self, cluster_id):
        parser.add_argument('posts', action='append')
        post_data = parser.parse_args()
        for one_post in post_data['posts']:
            dict_post = eval(one_post)
            print(dict_post['esId'])
            es.update(index='post_actions_'+cluster_id,id=dict_post['esId'],body={'doc':dict_post,'doc_as_upsert':True})
        return jsonify({"ok":"true"})

