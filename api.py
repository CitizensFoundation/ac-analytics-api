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

if os.environ.get('AC_SIMILARITY_API_URL'):
    api_url = os.environ['AC_SIMILARITY_API_URL']
else:
    api_url = '/api/v1'

es_url = os.environ['AC_SIMILARITY_ES_URL'] if os.environ.get('AC_SIMILARITY_ES_URL')!=None else 'localhost:9200'

master_api_key = os.environ['AC_SIMILARITY_MASTER_API_KEY']

#DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC=24*60*60
#COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC=1*60*60
#GROUP_TRIGGER_DEBOUNCE_TIME_SEC=10*60
#ARTICLES_TRIGGER_DEBOUNCE_TIME_SEC=15*60

DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC=180
COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC=120
GROUP_TRIGGER_DEBOUNCE_TIME_SEC=60
ARTICLES_TRIGGER_DEBOUNCE_TIME_SEC=60

app = Flask(__name__)
CORS(app)
api = Api(app)

parser = reqparse.RequestParser()

es = Elasticsearch(es_url)
queue = Queue(connection=conn)

def convertToNumbersWhereNeeded(inDict):
    outDict = {}
    for name, value in inDict.items():
        if ("_id" in name):
            outDict[name]=int(value)
        elif ("counter" in name):
            if (value==None):
              outDict[name]=0
            else:
              outDict[name]=int(value)
        else:
            outDict[name]=value
    return outDict

@app.before_request
def before_request():
    headers = request.headers
    auth = headers.get("X-Api-Key")
    if (auth!=master_api_key):
        return jsonify({"message": "ERROR: Unauthorized"}), 401

class DomainList(Resource):
    def post(self, domain_id):
        print("Call for: POST /domains")
        parser.add_argument('name')
        parser.add_argument('language')
        rawPost = parser.parse_args()
        print(rawPost)
        es.update(index='domains',doc_type='domain',id=int(domain_id),body={'doc':rawPost,'doc_as_upsert':True})
        return json.dumps({"ok": True})

class CommunityList(Resource):
    def post(self, community_id):
        print("Call for: POST /communities")
        parser.add_argument('name')
        parser.add_argument('language')
        rawPost = parser.parse_args()
        print(rawPost)
        es.update(index='communities',doc_type='community',id=int(community_id),body={'doc':rawPost,'doc_as_upsert':True})
        return json.dumps({"ok": True})

class GroupList(Resource):
    def post(self, group_id):
        print("Call for: POST /groups")
        parser.add_argument('name')
        parser.add_argument('language')
        rawPost = parser.parse_args()
        print(rawPost)
        es.update(index='groups',doc_type='group',id=int(group_id),body={'doc':rawPost,'doc_as_upsert':True})
        return json.dumps({"ok": True})

class PolicyGameList(Resource):
    def post(self, policy_game_id):
        print("Call for: POST /policy_games")
        parser.add_argument('name')
        parser.add_argument('language')
        rawPost = parser.parse_args()
        print(rawPost)
        es.update(index='policy_games',doc_type='policyGame',id=int(policy_game_id),body={'doc':rawPost,'doc_as_upsert':True})
        return json.dumps({"ok": True})

class PostList(Resource):
    triggerPostDomainQueueTimer = {}
    triggerPostCommunityQueueTimer = {}
    triggerPostGroupQueueTimer = {}

    def addToPostTriggerQueue(self, domain_id, community_id, group_id):
        print("addToPostTriggerQueue")

        queue.enqueue_call(
            func=triggerPostTraining, args=("posts", {
                "domain_id": domain_id,
                "community_id": community_id,
                "group_id": group_id,
                }), result_ttl=1*60*60*1000)
        if (domain_id!=None):
            PostList.triggerPostDomainQueueTimer[domain_id]=None;

        if (community_id!=None):
            PostList.triggerPostCommunityQueueTimer[community_id]=None;

        if (group_id!=None):
            PostList.triggerPostGroupQueueTimer[group_id]=None;

    def post(self, post_id):
        print("Call for: POST /posts")
        parser.add_argument('name')
        parser.add_argument('description')
        parser.add_argument('domain_id')
        parser.add_argument('community_id')
        parser.add_argument('group_id')
        parser.add_argument('status')
        parser.add_argument('official_status')
        parser.add_argument('counter_endorsements_up')
        parser.add_argument('counter_endorsements_down')
        parser.add_argument('counter_points')
        parser.add_argument('counter_flags')
        parser.add_argument('language')
        rawPost = parser.parse_args()
        #print(rawPost)
        esPost = convertToNumbersWhereNeeded(rawPost)
        esPost["lemmatizedContent"]=getLemmatizedText(esPost["description"], esPost.get("language"))
        print(esPost.get('name'))
        es.update(index='posts',doc_type='post',id=post_id,body={'doc':esPost,'doc_as_upsert':True})
        if PostList.triggerPostDomainQueueTimer.get(rawPost.domain_id)==None:
            PostList.triggerPostDomainQueueTimer[rawPost.domain_id] = Timer(DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPostTriggerQueue, [rawPost.domain_id, None, None])
            PostList.triggerPostDomainQueueTimer[rawPost.domain_id].start()

        if PostList.triggerPostCommunityQueueTimer.get(rawPost.community_id)==None:
            PostList.triggerPostCommunityQueueTimer[rawPost.community_id] = Timer(COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPostTriggerQueue, [None, rawPost.community_id, None])
            PostList.triggerPostCommunityQueueTimer[rawPost.community_id].start()

        if PostList.triggerPostGroupQueueTimer.get(rawPost.group_id)==None:
            PostList.triggerPostGroupQueueTimer[rawPost.group_id] = Timer(GROUP_TRIGGER_DEBOUNCE_TIME_SEC,  self.addToPostTriggerQueue, [None, None, rawPost.group_id])
            PostList.triggerPostGroupQueueTimer[rawPost.group_id].start()

        return json.dumps({"ok": True})

class PointList(Resource):
    triggerPointDomainQueueTimer = {}
    triggerPointCommunityQueueTimer = {}
    triggerPointGroupQueueTimer = {}

    def addToPointTriggerQueue(self, domain_id, community_id, group_id):
        print("addToPointTriggerQueue")

        queue.enqueue_call(
            func=triggerPointTraining, args=("points", {
                "domain_id": domain_id,
                "community_id": community_id,
                "group_id": group_id,
                }), result_ttl=1*60*60*1000)
        if (domain_id!=None):
            PointList.triggerPointDomainQueueTimer[domain_id]=None;

        if (community_id!=None):
            PointList.triggerPointCommunityQueueTimer[community_id]=None;

        if (group_id!=None):
            PointList.triggerPointGroupQueueTimer[group_id]=None;

    def post(self, point_id):
        print("Call for: POST /points")
        parser.add_argument('content')
        parser.add_argument('post_id')
        parser.add_argument('domain_id')
        parser.add_argument('community_id')
        parser.add_argument('group_id')
        parser.add_argument('status')
        parser.add_argument('counter_quality_up')
        parser.add_argument('counter_quality_down')
        parser.add_argument('counter_flags')
        parser.add_argument('value')
        parser.add_argument('language')
        rawPoint = parser.parse_args()
        print(rawPoint)
        esPoint = convertToNumbersWhereNeeded(rawPoint)
        esPoint["lemmatizedContent"]=getLemmatizedText(esPoint["content"], esPoint.get("language"))
        es.update(index='points',doc_type='point',id=point_id,body={'doc':esPoint,'doc_as_upsert':True})
        if PointList.triggerPointDomainQueueTimer.get(rawPoint.domain_id)==None:
            PointList.triggerPointDomainQueueTimer[rawPoint.domain_id] = Timer(DOMAIN_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPointTriggerQueue, [rawPoint.domain_id, None, None])
            PointList.triggerPointDomainQueueTimer[rawPoint.domain_id].start()

        if PointList.triggerPointCommunityQueueTimer.get(rawPoint.community_id)==None:
            PointList.triggerPointCommunityQueueTimer[rawPoint.community_id] = Timer(COMMUNITY_TRIGGER_DEBOUNCE_TIME_SEC, self.addToPointTriggerQueue, [None, rawPoint.community_id, None])
            PointList.triggerPointCommunityQueueTimer[rawPoint.community_id].start()

        if PointList.triggerPointGroupQueueTimer.get(rawPoint.group_id)==None:
            PointList.triggerPointGroupQueueTimer[rawPoint.group_id] = Timer(GROUP_TRIGGER_DEBOUNCE_TIME_SEC,  self.addToPointTriggerQueue, [None, None, rawPoint.group_id])
            PointList.triggerPointGroupQueueTimer[rawPoint.group_id].start()

        return json.dumps({"ok": True})

class ArticleList(Resource):
    triggerArticleQueueTimer = None

    def addToArticleTriggerQueue(self):
        print("addToArticleTriggerQueue")
        queue.enqueue_call(func=triggerArticleTraining, args=("articles", {}), result_ttl=1*60*60*1000)
        ArticleList.triggerArticleQueueTimer=None;

    def post(self, article_id):
        print("Call for: POST /articles")
        parser.add_argument('content')
        parser.add_argument('language')
        parser.add_argument('status')
        rawArticle = parser.parse_args()
        print(rawArticle)
        esArticle = convertToNumbersWhereNeeded(rawArticle)
        esArticle["lemmatizedContent"]=getLemmatizedText(esArticle["content"], esArticle.get("language"))

        es.update(index='articles',doc_type='article',id=article_id,body={'doc':esArticle,'doc_as_upsert':True})
        if ArticleList.triggerArticleQueueTimer==None:
            ArticleList.triggerArticleDomainQueueTimer = Timer(ARTICLES_TRIGGER_DEBOUNCE_TIME_SEC, self.addToArticleTriggerQueue)
            ArticleList.triggerArticleDomainQueueTimer.start()

        return json.dumps({"ok": True})

class FindSimilarPosts(Resource):
    def post(self):
        print("Call for: POST /find_similar")
        parser.add_argument('content')
        parser.add_argument('language')
        parser.add_argument('domain_id')
        parser.add_argument('community_id')
        parser.add_argument('group_id')
        rawFind = parser.parse_args()
        print(rawFind)
        esFind = convertToNumbersWhereNeeded(rawFind)
        language = esFind.get("language")
        lemmatizedContent=getLemmatizedText(esFind["content"],language)
        postSimilarity = PostSimilarity()

        similar_content = postSimilarity.getSimilarContentPost(lemmatizedContent, language, rawFind)

        return json.dumps(similar_content)

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

api.add_resource(PostList, api_url+'/posts/<post_id>')
api.add_resource(PointList, api_url+'/points/<point_id>')
api.add_resource(ArticleList, api_url+'/articles/<article_id>')
api.add_resource(DomainList, api_url+'/domains/<domain_id>')
api.add_resource(CommunityList, api_url+'/communities/<community_id>')
api.add_resource(GroupList, api_url+'/groups/<group_id>')
api.add_resource(PolicyGameList, api_url+'/policy_games/<policy_game_id>')
api.add_resource(FindSimilarPosts, api_url+'/find_similar_posts')
app.run()
