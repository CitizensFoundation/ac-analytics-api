from elasticsearch import Elasticsearch
from train_d2v import TrainDoc2Vec
from training_prefix import makeTrainingPrefix
import os

es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get('AC_SIM_ES_URL')!=None else 'localhost:9200'
es = Elasticsearch(es_url)

class Trainer:
  def __init__(self, index, docType, object):
    self.index = index
    self.docType = docType
    self.object = object
    self.filename_prefix = makeTrainingPrefix(index, object)
    if (object.get("domain_id")!=None):
      self.searchTerms = {"domain_id": int(object["domain_id"])}
    elif (object["community_id"] != None):
      self.searchTerms = {"community_id": int(object["community_id"])}
    elif (object["group_id"]):
      self.searchTerms = {"group_id": int(object["group_id"])}
    else:
      self.searchTerms = {}

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

    return es.search(index=self.index, body=body)

  def getAllPostTextFromES(self):
    items = self.getAllItemsFromES()
    outItemTexts = []
    outItemIds = []
    for item in items['hits']['hits']:
      itemText = ""
      if item["_source"]["name"]:
        itemText+=item["_source"]["name"]+" "
      itemText+=item["_source"]["description"]+"\n"
      outItemTexts.append(itemText)
      outItemIds.append(item["_id"])
    print(outItemTexts)
    print(outItemIds)
    return [outItemTexts, outItemIds]

  def start(self):
    texts, ids = self.getAllPostTextFromES()
    print(texts)
    print(ids)
    d2v = TrainDoc2Vec(self.filename_prefix, texts, ids)
    d2v.train()
    print("Training done for: "+self.filename_prefix)

def triggerPostTraining(type, object):
  print("triggerPostTraining")
  print(object)
  trainer = Trainer("posts","post",object)
  trainer.start()

def triggerPointTraining(type, object):
  print("triggerPointTraining")
  print(object)
  trainer = Trainer("points","post",object)
  trainer.start()

def triggerArticleTraining(type, object):
  print("triggerArticleTraining")
  print(object)
  trainer = Trainer("articles","article",object)
  trainer.start()
