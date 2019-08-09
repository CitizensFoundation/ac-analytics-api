from elasticsearch import Elasticsearch
from train_w2v import TrainWord2Vec

es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get('AC_SIM_ES_URL')!=None else 'localhost:9200'
es = Elasticsearch(es_url)

class Trainer:
  def __init__(index, docType, object):
    self.index = index
    self.docType = docType
    self.object = object
    self.filename_prefix = index+"_"
    if (object.get("domain_id")!=None):
      self.searchTerms = {"domain_id": int(object["domain_id"])}
      self.filename_prefix = object["domain_id"]+"_"+"_"
    elif (object["community_id"] != None):
      self.searchTerms = {"community_id": int(object["community_id"])}
      self.filename_prefix = "_"+object["community_id"]+"_"
    elif (object["group_id"]):
      self.searchTerms = {"group_id": int(object["group_id"])}
      self.filename_prefix = "_"+"_"+object["group_id"]
    else:
      self.searchTerms = {}
      self.filename_prefix = "_"+"_"+"article_"

  def getAllItemsFromES(self):
    body = {
        "query": {
            "must": {
                "term":  self.searchTerms
          }
        }
    }

    return es.search(index=self.index, doc_type=self.docType, body=body)

  def getAllPostTextFromES(self):
    items = self.getAllItemsFromES()
    itemText = ""
    for item in items:
      if item.name:
        itemText+=item.name+" "
      itemText+=item.description+"\n"
    return itemText

  def start(self):
    w2v = TrainWord2Vec(self.filename_prefix, getAllPostTextFromES())
    w2v.train()
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
