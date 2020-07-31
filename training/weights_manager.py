from elasticsearch import Elasticsearch
from training.training_prefix import makeTrainingPrefix
from gensim.models.doc2vec import Doc2Vec
import json
import os
import traceback

MAX_NUMBER_OF_SIMILAR_DOCUMENTS=20
#CUTTOFF_FOR_WEIGTHS=0.75
CUTTOFF_FOR_WEIGTHS=0.33
CUTTOFF_FOR_SAVING_WEIGTHS=0.10

es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get('AC_SIM_ES_URL')!=None else 'localhost:9200'
es = Elasticsearch(es_url)

class WeightsManager:
  def __init__(self, indexName, docType, object, model):
    self.indexName = indexName
    self.docType = docType
    self.object = object
    if (object.get("domain_id")!=None):
      self.searchTerms = {"domain_id": int(object["domain_id"])}
      self.collectionIndexName = "domains_"+object["cluster_id"]
      self.colletionIndexDocType = "domain"
      self.collectionIndexSearchId = int(object["domain_id"])
    elif (object.get("community_id") != None):
      self.searchTerms = {"community_id": int(object["community_id"])}
      self.collectionIndexName = "communities_"+object["cluster_id"]
      self.colletionIndexDocType = "community"
      self.collectionIndexSearchId = int(object["community_id"])
    elif (object.get("group_id")):
      self.searchTerms = {"group_id": int(object["group_id"])}
      self.collectionIndexName = "groups_"+object["cluster_id"]
      self.colletionIndexDocType = "group"
      self.collectionIndexSearchId = int(object["group_id"])
    elif (object.get("post_id")):
      self.searchTerms = {"post_id": int(object["post_id"])}
      self.collectionIndexName = "posts_"+object["cluster_id"]
      self.colletionIndexDocType = "post"
      self.collectionIndexSearchId = int(object["post_id"])
    elif (object.get("policy_game_id")):
      self.searchTerms = {"policy_game_id": int(object["policy_game_id"])}
      self.collectionIndexName = "policy_games_"+object["cluster_id"]
      self.colletionIndexDocType = "policy_game"
      self.collectionIndexSearchId = int(object["policy_game_id"])

    res = es.get(index=self.collectionIndexName,  id=self.collectionIndexSearchId)
    self.language = res['_source']['language']
    self.weightIndexType = "weights_"+makeTrainingPrefix(self.language, indexName, object)
    self.modelFilePrefix = makeTrainingPrefix(self.language, indexName, object)
    filename = "d2v_models/"+self.modelFilePrefix+'_d2v.model'
    self.model = model

  def getAllItemsFromES(self):
    body = {
        "query": {
          "bool": {
            "must": {
                "term":  self.searchTerms
            }
          }
        }
    }

    #TODO add a scroll here to be able to process more than 10.000
    return es.search(index=self.indexName, body=body, size=10*1000)

  def getAllIdsFromES(self):
    items = self.getAllItemsFromES()
    outItemIds = []
    for item in items['hits']['hits']:
      outItemIds.append(int(item["_id"]))
    return outItemIds

  def getAllWeightsFromES(self):
    indexTypeDict = {"term": {"indexType": self.weightIndexType } }
    #moreThanLimit = [{"range":{"value": {"gte":CUTTOFF_FOR_WEIGTHS }}},{"range":{"value": {"lte":0.75 }}}]
    moreThanLimit = [{"range":{"value": {"gte":CUTTOFF_FOR_WEIGTHS }}}]
    body = {
        "query": {
          "bool": {
            "filter": moreThanLimit,
            "must": [
              indexTypeDict
            ]
          }
        }
    }
    #TODO A cursor for larger results
    print(moreThanLimit)
    print(indexTypeDict)
    print(self.object["cluster_id"])
    return es.search(index="similarityweights_"+self.object["cluster_id"], body=body, size=10*1000)

  def deleteWeightsFromES(self):
    body = {
        "query": {
          "bool": {
            "must": {
                "term":  {"indexType": self.weightIndexType }
            }
          }
        }
    }

    if es.indices.exists("similarityweights_"+self.object["cluster_id"]):
      res = es.delete_by_query(index="similarityweights_"+self.object["cluster_id"], body=body, size=10*1000)
      print("Deleted similarityweights: "+self.weightIndexType)

  def processSimilarity(self, textId):
    #print("MOST SIMILAR FOR: "+str(textId))

    #TODO: Confirm if this is needed https://github.com/RaRe-Technologies/gensim/issues/2260
    #self.model.docvecs.vectors_docs_norm = None
    #self.model.docvecs.init_sims()
    try:
      most_similar = self.model.docvecs.most_similar([str(textId)], topn = MAX_NUMBER_OF_SIMILAR_DOCUMENTS)
    except Exception as e:
      print("DOC2VEC: error")
      print(e)
      return
    #print(most_similar)
    for similarId,similarWeight in most_similar:
      #print("MOST sim id: "+similarId)
      #print("MOST w: "+str(similarWeight))
      if float(similarWeight)>CUTTOFF_FOR_SAVING_WEIGTHS:
        if int(textId)<=int(similarId):
          source=textId
          target=similarId
        else:
          source=similarId
          target=textId
        body = {
          "source": source,
          "target": target,
          "value":  similarWeight,
          "indexType": self.weightIndexType
        }
        id=source+"_"+target+"_"+self.weightIndexType
        #print("Saved item with weight: "+str(similarWeight))
        es.update(index="similarityweights_"+self.object["cluster_id"],id=id,body={'doc':body,'doc_as_upsert':True})
      else:
        print("Item not saved with low weight: "+str(similarWeight))

  def countLinks(self, links, nodeId):
    count = 0
    for link in links:
      if (link["source"]==nodeId):
        count+=1
      if (link["target"]==nodeId):
        count+=1
    return count

  def getNodesAndLinksFromES(self):
    nodes = self.getAllItemsFromES()["hits"]["hits"]
    links = self.getAllWeightsFromES()["hits"]["hits"]

    outLinks = []
    for link in links:
      if (link["_source"]["value"]>CUTTOFF_FOR_WEIGTHS):
        outLinks.append(link["_source"])

    outNodes = []
    for node in nodes:
      outNodes.append(
        {
          "id": node["_id"],
          "group": node["_source"].get("group_id"),
          "postId": node["_source"].get("post_id"),
          "name": node["_source"].get("name"),
          "content": node["_source"].get("content"),
          "imageUrl": node["_source"].get("imageUrl"),
          "counter_endorsements_down": node["_source"].get("counter_endorsements_down"),
          "counter_endorsements_up": node["_source"].get("counter_endorsements_up"),
          "linkCount": self.countLinks(outLinks, node["_id"]),
          "lemmatizedContent": node["_source"].get("lemmatizedContent")
        }
      )
    return {"nodes": outNodes, "links": outLinks }

  def startProcessing(self):
    self.deleteWeightsFromES()
    ids = self.getAllIdsFromES()
    i=0
    for id in ids:
      try:
        self.processSimilarity(str(id))
      except Exception as e:
        print("ERROR: processSimilarity for "+str(id))
        print(e)
        traceback.print_exc()
        pass
      i+=1
    print("Completed create weights")
