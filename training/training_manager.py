from elasticsearch import Elasticsearch
from training.train_d2v import TrainDoc2Vec
from training.training_prefix import makeTrainingPrefix
import os

es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get('AC_SIM_ES_URL')!=None else 'localhost:9200'
es = Elasticsearch(es_url)

class TrainingManager:
  def __init__(self, indexName, docType, object):
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
      self.collectionIndexName = "policy_games"
      self.colletionIndexDocType = "policy_game"
      self.collectionIndexSearchId = int(object["policy_game_id"])

    res = es.get(index=self.collectionIndexName, id=self.collectionIndexSearchId)
    #print(res['_source'])
    self.filename_prefix = makeTrainingPrefix(res['_source']['language'], indexName, object)

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

    #print("TERMS")
    #print(body)
    #print(self.docType)

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
        outItemTexts.append(itemText)
        outItemIds.append(str(item["_id"]))
        #print("SCORE: "+item["_source"].get("lemmatizedContent"))
      else:
        print("Warning: did not find lemmatizedContent for item")
    #print(outItemTexts)
    #print(outItemIds)
    return [outItemTexts, outItemIds]

  def start(self):
    texts, ids = self.getAllTextFromES()
    print("TEXTS")
    print(texts)
    print("IDS")
    print(ids)
    if ids and texts and len(ids)>0 and len(texts)>0:
      d2v = TrainDoc2Vec(self.filename_prefix, texts, ids)
      d2v.train()
      self.model = d2v.model
      print("Training done for: "+self.filename_prefix)
      return True
    else:
      print("Warning: no texts for training, skipping")
      return False