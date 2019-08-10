from elasticsearch import Elasticsearch
from training.training_prefix import makeTrainingPrefix
from gensim.models.doc2vec import Doc2Vec
import json
import os

es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get('AC_SIM_ES_URL')!=None else 'localhost:9200'
es = Elasticsearch(es_url)

class CreateWeights:
  def __init__(self, indexName, docType, object, model):
    self.indexName = indexName
    self.docType = docType
    self.object = object
    if (object.get("domain_id")!=None):
      self.searchTerms = {"domain_id": int(object["domain_id"])}
      self.collectionIndexName = "domains"
      self.colletionIndexDocType = "domain"
      self.indexSearchId = int(object["domain_id"])
    elif (object["community_id"] != None):
      self.searchTerms = {"community_id": int(object["community_id"])}
      self.collectionIndexName = "communities"
      self.colletionIndexDocType = "community"
      self.indexSearchId = int(object["community_id"])
    elif (object["group_id"]):
      self.searchTerms = {"group_id": int(object["group_id"])}
      self.collectionIndexName = "groups"
      self.colletionIndexDocType = "group"
      self.indexSearchId = int(object["group_id"])
    elif (object["policy_game_id"]):
      self.searchTerms = {"policy_game_id": int(object["policy_game_id"])}
      self.collectionIndexName = "policy_games"
      self.colletionIndexDocType = "policy_game"
      self.indexSearchId = int(object["policy_game_id"])

    res = es.get(index=self.collectionIndexName, doc_type=self.colletionIndexDocType, id=1)
    print(res['_source'])
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

    print("TERMS")
    print(body)
    print(self.docType)

    #TODO add a scroll here to be able to process more than 10.000
    return es.search(index=self.indexName, body=body, size=10*1000)

  def getAllTextFromES(self):
    items = self.getAllItemsFromES()
    #print(items['hits']['hits'])
    outItemTexts = []
    outItemIds = []
    for item in items['hits']['hits']:
      itemText = ""
      if (item["_source"].get("lemmatizedContent")):
        itemText+=item["_source"].get("lemmatizedContent")
      else:
        print("ERROR: did not find lemmatized text for item")
      outItemTexts.append(itemText)
      outItemIds.append(int(item["_id"]))
    return [outItemTexts, outItemIds]

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

    if es.indices.exists("similarityWeights"):
      res = es.delete_by_query(index="similarityWeights", body=body, size=10*1000)
      print("DELETE: "+body)
      print(res)

  def processSimilarity(self, textId, text):
    print("MOST SIMILAR FOR: "+str(textId))
    print("TEXT TO CHECK: "+text)
    most_similar = self.model.docvecs.most_similar([str(textId)], topn = 5)
    print(most_similar)

  def start(self):
    self.deleteWeightsFromES()
    texts, ids = self.getAllTextFromES()
    i=0
    for text in texts:
        print("ID: "+str(ids[i]))
        self.processSimilarity(str(ids[i]), texts[i])
        i+=1
