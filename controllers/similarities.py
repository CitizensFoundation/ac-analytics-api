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
import os.path
from os import path

import json
import sys

from threading import Timer

from flask import jsonify
from flask_restful import reqparse, Resource

from rq import Queue
from rq.job import Job

from elasticsearch import Elasticsearch, NotFoundError

from training.training import triggerPostTraining, triggerPointTraining, triggerArticleTraining
from worker import conn
from lemmatizer.lemmatizer import getLemmatizedText
from similarities.similarities import PostSimilarity
from training.weights_manager import WeightsManager

if os.environ.get('AC_ANALYTICS_API_URL'):
    api_url = os.environ['AC_ANALYTICS_API_URL']
else:
    api_url = '/api/v1'

es_url = os.environ['AC_ANALYTICS_ES_URL'] if os.environ.get('AC_ANALYTICS_ES_URL')!=None else 'localhost:9200'

master_api_key = os.environ['AC_ANALYTICS_MASTER_API_KEY']

MIN_CHARACTER_LENGTH_FOR_PROCESSING=15
MIN_CHARACTER_LENGTH_FOR_POINT_PROCESSING=15

#DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC=24*60*60
#COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC=1*60*60
#GROUP_TRIGGER_DEBOUNCE_TIME_SEC=10*60
#ARTICLES_TRIGGER_DEBOUNCE_TIME_SEC=15*60

DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC=60*45
COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC=60*15
GROUP_TRIGGER_DEBOUNCE_TIME_SEC=3*60
POST_TRIGGER_DEBOUNCE_TIME_SEC=3*60
ARTICLES_TRIGGER_DEBOUNCE_TIME_SEC=60

es = Elasticsearch(es_url)
queue = Queue(connection=conn, default_timeout=6000)

def convertToNumbersWhereNeeded(inDict):
    outDict = {}
    for name, value in inDict.items():
        if ("_id" in name and value!=None):
            outDict[name]=int(value)
        elif ("counter" in name):
            if (value==None):
              outDict[name]=0
            else:
              outDict[name]=int(value)
        else:
            outDict[name]=value
    return outDict

class DomainList(Resource):
    def post(self, cluster_id, domain_id):
        print("Call for: POST /domains")
        parser = reqparse.RequestParser()
        parser.add_argument('name')
        parser.add_argument('description')
        parser.add_argument('status')
        parser.add_argument('language')
        parser.add_argument('created_at')
        parser.add_argument('updated_at')
        rawPost = parser.parse_args()
        language = rawPost['language'][:2]
        language = language.lower()
        rawPost['language']=language
        print(rawPost)
        if rawPost['status']=='published':
            es.update(index='domains_'+cluster_id,id=int(domain_id),body={'doc':rawPost,'doc_as_upsert':True})
        else:
            try:
                es.delete(index='domains_'+cluster_id,id=int(domain_id))
            except NotFoundError:
                print("Domain not found for delete: "+domain_id)
                pass

        return json.dumps({"ok": True})

class CommunityList(Resource):
    def post(self, cluster_id, community_id):
        print("Call for: POST /communities")
        parser = reqparse.RequestParser()
        parser.add_argument('name')
        parser.add_argument('description')
        parser.add_argument('language')
        parser.add_argument('status')
        parser.add_argument('created_at')
        parser.add_argument('updated_at')
        rawPost = parser.parse_args()
        language = rawPost['language'][:2]
        language = language.lower()
        rawPost['language']=language
        print(rawPost)
        if rawPost['status']=='published':
            es.update(index='communities_'+cluster_id,id=int(community_id),body={'doc':rawPost,'doc_as_upsert':True})
        else:
            try:
                es.delete(index='communities_'+cluster_id,id=int(community_id))
            except NotFoundError:
                print("Community not found for delete: "+community_id)
                pass
        return json.dumps({"ok": True})

class GroupList(Resource):
    def post(self, cluster_id, group_id):
        print("Call for: POST /groups")
        parser = reqparse.RequestParser()
        parser.add_argument('name')
        parser.add_argument('objectives')
        parser.add_argument('status')
        parser.add_argument('language')
        parser.add_argument('created_at')
        parser.add_argument('updated_at')
        rawPost = parser.parse_args()
        language = rawPost['language'][:2]
        language = language.lower()
        rawPost['language']=language

        print(rawPost)

        if rawPost['status']=='published':
            es.update(index='groups_'+cluster_id,id=int(group_id),body={'doc':rawPost,'doc_as_upsert':True})
        else:
            try:
                es.delete(index='groups_'+cluster_id,id=int(group_id))
            except NotFoundError:
                print("Group not found for delete: "+group_id)
                pass

        return json.dumps({"ok": True})

#TODO: Make sure first to do toxicity check and language before triggering this as the group language might be wrong
class PostList(Resource):
    triggerPostDomainQueueTimer = {}
    triggerPostCommunityQueueTimer = {}
    triggerPostGroupQueueTimer = {}

    def addToPostTriggerQueue(self, cluster_id, domain_id, community_id, group_id):
        print("addToPostTriggerQueue")

        lockFilename = "/tmp/acaRqInQueuePosts_{}_{}_{}_{}".format(cluster_id, domain_id, community_id, group_id);

        if path.exists(lockFilename):
            print("Already in queue: "+lockFilename)
        else:
            f = open(lockFilename, "w")
            f.write("x")
            f.close()

            queue.enqueue_call(
                func=triggerPostTraining, args=("posts", {
                    "cluster_id": cluster_id,
                    "domain_id": domain_id,
                    "community_id": community_id,
                    "group_id": group_id,
                    "lockFilename": lockFilename
                    }), result_ttl=1*60*60*1000)

            if (domain_id!=None):
                PostList.triggerPostDomainQueueTimer[domain_id]=None;

            if (community_id!=None):
                PostList.triggerPostCommunityQueueTimer[community_id]=None;

            if (group_id!=None):
                PostList.triggerPostGroupQueueTimer[group_id]=None;

    def triggerTrainingUpdate(self, cluster_id, rawPost):
        if PostList.triggerPostDomainQueueTimer.get(rawPost.domain_id)==None:
            PostList.triggerPostDomainQueueTimer[rawPost.domain_id] = Timer(DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPostTriggerQueue, [cluster_id, rawPost.domain_id, None, None])
            PostList.triggerPostDomainQueueTimer[rawPost.domain_id].start()

        if PostList.triggerPostCommunityQueueTimer.get(rawPost.community_id)==None:
            PostList.triggerPostCommunityQueueTimer[rawPost.community_id] = Timer(COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPostTriggerQueue, [cluster_id, None, rawPost.community_id, None])
            PostList.triggerPostCommunityQueueTimer[rawPost.community_id].start()

        if PostList.triggerPostGroupQueueTimer.get(rawPost.group_id)==None:
            PostList.triggerPostGroupQueueTimer[rawPost.group_id] = Timer(GROUP_TRIGGER_DEBOUNCE_TIME_SEC,  self.addToPostTriggerQueue, [cluster_id, None, None, rawPost.group_id])
            PostList.triggerPostGroupQueueTimer[rawPost.group_id].start()

    def post(self, cluster_id, post_id):
        print("Call for: POST /posts")
        parser = reqparse.RequestParser()
        parser.add_argument('name')
        parser.add_argument('description')
        parser.add_argument('domain_id')
        parser.add_argument('community_id')
        parser.add_argument('group_id')
        parser.add_argument('user_id')
        parser.add_argument('status')
        parser.add_argument('official_status')
        parser.add_argument('counter_endorsements_up')
        parser.add_argument('counter_endorsements_down')
        parser.add_argument('counter_points')
        parser.add_argument('counter_flags')
        parser.add_argument('imageUrl')
        parser.add_argument('created_at')
        parser.add_argument('updated_at')
        parser.add_argument('videoUrl')
        parser.add_argument('audioUrl')
        parser.add_argument('publicAccess')
        parser.add_argument('communityAccess')
        parser.add_argument('language')
        rawPost = parser.parse_args()
        language = rawPost['language'][:2]
        language = language.lower()
        rawPost['language']=language
        #print(rawPost)
        if rawPost['status']=='published':
            esPost = convertToNumbersWhereNeeded(rawPost)
            if (len(rawPost.get("description"))>MIN_CHARACTER_LENGTH_FOR_PROCESSING):
                print("Post len: "+str(len(rawPost.get("description")))+" words: "+str(len(rawPost.get("description").split())))
                esPost["lemmatizedContent"]=getLemmatizedText(esPost["name"], esPost["description"], esPost.get("language"))
            else:
                esPost['tooShort']=True
                print("Warning: POST TOO SHORT FOR PROCESSING - min chars: "+str(MIN_CHARACTER_LENGTH_FOR_PROCESSING)+ " current: "+str(len(rawPost.get("description"))))

            if (esPost.get("lemmatizedContent")!=None and len(esPost.get("lemmatizedContent"))>0):
                self.triggerTrainingUpdate(cluster_id, rawPost)
            else:
                print("Warning: NO DESCRIPTION FOR POST")
            es.update(index='posts_'+cluster_id,id=post_id,body={'doc':esPost,'doc_as_upsert':True})
        else:
            try:
                es.delete(index='posts_'+cluster_id,id=post_id)
            except NotFoundError:
                print("Post not found for delete: "+post_id)
                pass
            self.triggerTrainingUpdate(cluster_id, rawPost)
        return json.dumps({"ok": True})

class PointList(Resource):
    triggerPointDomainQueueTimer = {}
    triggerPointCommunityQueueTimer = {}
    triggerPointGroupQueueTimer = {}
    triggerPointPostQueueTimer = {}

    def addToPointTriggerQueue(self, cluster_id, domain_id, community_id, group_id, post_id):
        print("addToPointTriggerQueue")

        lockFilename = "/tmp/acaRqInQueuePosts_{}_{}_{}_{}_{}".format(cluster_id, domain_id, community_id, group_id, post_id);

        if path.exists(lockFilename):
            print("Already in queue: "+lockFilename)
        else:
            f = open(lockFilename, "w")
            f.write("x")
            f.close()

            queue.enqueue_call(
                func=triggerPointTraining, args=("points", {
                    "cluster_id": cluster_id,
                    "domain_id": domain_id,
                    "community_id": community_id,
                    "group_id": group_id,
                    "post_id": post_id,
                    "lockFilename": lockFilename
                    }), result_ttl=1*60*60*1000)
            if (domain_id!=None):
                PointList.triggerPointDomainQueueTimer[domain_id]=None;

            if (community_id!=None):
                PointList.triggerPointCommunityQueueTimer[community_id]=None;

            if (group_id!=None):
                PointList.triggerPointGroupQueueTimer[group_id]=None;

            if (post_id!=None):
                PointList.triggerPointPostQueueTimer[post_id]=None;

    def triggerPointTrainingUpdate(self, cluster_id, rawPoint):
        print("Triggering triggerPointTrainingUpdate")
        if PointList.triggerPointDomainQueueTimer.get(rawPoint.domain_id)==None:
            PointList.triggerPointDomainQueueTimer[rawPoint.domain_id] = Timer(DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPointTriggerQueue, [cluster_id, rawPoint.domain_id, None, None, None])
            PointList.triggerPointDomainQueueTimer[rawPoint.domain_id].start()

        if PointList.triggerPointCommunityQueueTimer.get(rawPoint.community_id)==None:
            PointList.triggerPointCommunityQueueTimer[rawPoint.community_id] = Timer(COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPointTriggerQueue, [cluster_id, None, rawPoint.community_id, None, None])
            PointList.triggerPointCommunityQueueTimer[rawPoint.community_id].start()

        if PointList.triggerPointGroupQueueTimer.get(rawPoint.group_id)==None:
            PointList.triggerPointGroupQueueTimer[rawPoint.group_id] = Timer(GROUP_TRIGGER_DEBOUNCE_TIME_SEC,  self.addToPointTriggerQueue, [cluster_id, None, None, rawPoint.group_id, None])
            PointList.triggerPointGroupQueueTimer[rawPoint.group_id].start()

        if PointList.triggerPointPostQueueTimer.get(rawPoint.post_id)==None:
            PointList.triggerPointPostQueueTimer[rawPoint.post_id] = Timer(POST_TRIGGER_DEBOUNCE_TIME_SEC,  self.addToPointTriggerQueue, [cluster_id, None, None, None, rawPoint.post_id])
            PointList.triggerPointPostQueueTimer[rawPoint.post_id].start()

    def post(self, cluster_id, point_id):
        parser = reqparse.RequestParser()
        parser.add_argument('content')
        parser.add_argument('post_id')
        parser.add_argument('domain_id')
        parser.add_argument('community_id')
        parser.add_argument('group_id')
        parser.add_argument('user_id')
        parser.add_argument('status')
        parser.add_argument('post_status')
        parser.add_argument('publicAccess')
        parser.add_argument('communityAccess')
        parser.add_argument('videoUrl')
        parser.add_argument('audioUrl')
        parser.add_argument('counter_quality_up')
        parser.add_argument('counter_quality_down')
        parser.add_argument('counter_flags')
        parser.add_argument('value')
        parser.add_argument('language')
        rawPoint = parser.parse_args()
        #print(rawPoint)
        language = rawPoint['language'][:2]
        language = language.lower()
        rawPoint['language']=language

        if rawPoint['status']=='published' and rawPoint['post_status']=='published':
            esPoint = convertToNumbersWhereNeeded(rawPoint)

            if (len(esPoint.get("content"))>MIN_CHARACTER_LENGTH_FOR_POINT_PROCESSING):
                print("Point len: "+str(len(esPoint.get("content")))+" words: "+str(len(esPoint.get("content").split())))
                esPoint["lemmatizedContent"]=getLemmatizedText("", esPoint["content"], esPoint.get("language"))
            else:
                esPoint['tooShort']=True
                print("Warning: POINT TOO SHORT FOR PROCESSING - min chars: "+str(MIN_CHARACTER_LENGTH_FOR_POINT_PROCESSING)+ " current: "+str(len(esPoint.get("content"))))

            if (esPoint.get("lemmatizedContent")!=None and len(esPoint.get("lemmatizedContent"))>0):
                self.triggerPointTrainingUpdate(cluster_id, rawPoint)
            else:
                esPoint['noLemmatizedContent']=True
                print("Warning: NO CONTENT FOR POINT")

            es.update(index='points_'+cluster_id,id=point_id,body={'doc':esPoint,'doc_as_upsert':True})
        else:
            try:
                es.delete(index='points_'+cluster_id,id=point_id)
            except NotFoundError:
                print("Point not found for delete: "+point_id)
                pass
            self.triggerPointTrainingUpdate(cluster_id, rawPoint)

        return json.dumps({"ok": True})

class FindSimilarPosts(Resource):
    def post(self, cluster_id):
        print("Call for: POST /find_similar")
        parser = reqparse.RequestParser()
        parser.add_argument('content')
        parser.add_argument('name')
        parser.add_argument('language')
        parser.add_argument('domain_id')
        parser.add_argument('community_id')
        parser.add_argument('cluster_id')
        parser.add_argument('group_id')
        rawFind = parser.parse_args()
        print(rawFind)
        esFind = convertToNumbersWhereNeeded(rawFind)
        language = esFind.get("language")
        lemmatizedContent=getLemmatizedText(esFind["name"], esFind["content"],language)
        postSimilarity = PostSimilarity()

        similar_content = postSimilarity.getSimilarContentPost(lemmatizedContent, language, rawFind)

        return json.dumps(similar_content)

class GetCommunityPostsWithWeights(Resource):
    def get(self, cluster_id, community_id):
        weights = WeightsManager("posts_"+cluster_id,"post",{"community_id": community_id, "cluster_id": cluster_id}, None)
        return jsonify(weights.getNodesAndLinksFromES())

class GetGroupPostsWithWeights(Resource):
    def get(self, cluster_id, group_id):
        weights = WeightsManager("posts_"+cluster_id,"post",{"group_id": group_id, "cluster_id": cluster_id}, None)
        return jsonify(weights.getNodesAndLinksFromES())

class GetDomainPostsWithWeights(Resource):
    def get(self, cluster_id, domain_id):
        weights = WeightsManager("posts_"+cluster_id,"post",{"domain_id": domain_id, "cluster_id": cluster_id}, None)
        return jsonify(weights.getNodesAndLinksFromES())

class GetPostPointsWithWeights(Resource):
    def get(self, cluster_id, post_id):
        weights = WeightsManager("points_"+cluster_id,"point",{"post_id": post_id, "cluster_id": cluster_id}, None)
        return jsonify(weights.getNodesAndLinksFromES())

class TriggerDomainPostTraining(Resource):
    def put(self, cluster_id, domain_id):
        queue.enqueue_call(
            func=triggerPostTraining, args=("posts", {
                "cluster_id": cluster_id,
                "domain_id": domain_id,
                "community_id": None,
                "group_id": None,
                }), result_ttl=1*60*60*1000, timeout=6000)

class TriggerCommunityPostTraining(Resource):
    def put(self, cluster_id, community_id):
        queue.enqueue_call(
            func=triggerPostTraining, args=("posts", {
                "cluster_id": cluster_id,
                "domain_id": None,
                "community_id": community_id,
                "group_id": None,
                }), result_ttl=1*60*60*1000, timeout=6000)

class TriggerGroupPostTraining(Resource):
    def put(self, cluster_id, group_id):
        queue.enqueue_call(
            func=triggerPostTraining, args=("posts", {
                "cluster_id": cluster_id,
                "domain_id": None,
                "community_id": None,
                "group_id": group_id,
                }), result_ttl=1*60*60*1000, timeout=6000)

class TriggerPostPointsTraining(Resource):
    def put(self, cluster_id, post_id):
        queue.enqueue_call(
            func=triggerPostTraining, args=("points", {
                "cluster_id": cluster_id,
                "domain_id": None,
                "community_id": None,
                "group_id": None,
                "post_id": post_id
                }), result_ttl=1*60*60*1000, timeout=6000)