from elasticsearch import Elasticsearch
from training.train_d2v import TrainDoc2Vec
from training.training_prefix import makeTrainingPrefix
import os
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

es_url = os.environ['AC_SIM_ES_URL'] if os.environ.get('AC_SIM_ES_URL')!=None else 'localhost:9200'
es = Elasticsearch(es_url)

icelandic_stop_words = ["það","þetta","það","eru","ekki",",",",","sér","sig","sín","aðra","aðrar","aðrir","alla","allan","allar","allir","allnokkra","allnokkrar","allnokkrir","allnokkru","allnokkrum","allnokkuð","allnokkur","allnokkurn","allnokkurra","allnokkurrar","allnokkurri","allnokkurs","allnokkurt","allra","allrar","allri","alls","allt","allur","annað","annan","annar","annarra","annarrar","annarri","annars","báða","báðar","báðir","báðum","bæði","beggja","ein","eina","einar","einhver","einhverja","einhverjar","einhverjir","einhverju","einhverjum","einhvern","einhverra","einhverrar","einhverri","einhvers","einir","einn","einna","einnar","einni","eins","einskis","einu","einum","eitt","eitthvað","eitthvert","ekkert","enga","engan","engar","engin","enginn","engir","engra","engrar","engri","engu","engum","fáein","fáeina","fáeinar","fáeinir","fáeinna","fáeinum","flestalla","flestallan","flestallar","flestallir","flestallra","flestallrar","flestallri","flestalls","flestallt","flestallur","flestöll","flestöllu","flestöllum","hin","hina","hinar","hinir","hinn","hinna","hinnar","hinni","hins","hinu","hinum","hitt","hvað","hvaða","hver","hverja","hverjar","hverjir","hverju","hverjum","hvern","hverra","hverrar","hverri","hvers","hvert","hvílík","hvílíka","hvílíkan","hvílíkar","hvílíkir","hvílíkra","hvílíkrar","hvílíkri","hvílíks","hvílíkt","hvílíku","hvílíkum","hvílíkur","hvor","hvora","hvorar","hvorir","hvorn","hvorra","hvorrar","hvorri","hvors","hvort","hvoru","hvorug","hvoruga","hvorugan","hvorugar","hvorugir","hvorugra","hvorugrar","hvorugri","hvorugs","hvorugt","hvorugu","hvorugum","hvorugur","hvorum","mestalla","mestallan","mestallar","mestallir","mestallra","mestallrar","mestallri","mestalls","mestallt","mestallur","mestöll","mestöllu","mestöllum","mín","mína","mínar","mínir","minn","minna","minnar","minni","míns","mínu","mínum","mitt","nein","neina","neinar","neinir","neinn","neinna","neinnar","neinni","neins","neinu","neinum","neitt","nokkra","nokkrar","nokkrir","nokkru","nokkrum","nokkuð","nokkur","nokkurn","nokkurra","nokkurrar","nokkurri","nokkurs","nokkurt","öðru","öðrum","öll","öllu","öllum","önnur","sá","sama","saman","samar","sami","samir","samra","samrar","samri","sams","samt","samur","sérhvað","sérhver","sérhverja","sérhverjar","sérhverjir","sérhverju","sérhverjum","sérhvern","sérhverra","sérhverrar","sérhverri","sérhvers","sérhvert","sín","sína","sínar","sínhver","sínhverja","sínhverjar","sínhverjir","sínhverju","sínhverjum","sínhvern","sínhverra","sínhverrar","sínhverri","sínhvers","sínhvert","sínhvor","sínhvora","sínhvorar","sínhvorir","sínhvorn","sínhvorra","sínhvorrar","sínhvorri","sínhvors","sínhvort","sínhvoru","sínhvorum","sínir","sinn","sinna","sinnar","sinnhver","sinnhverja","sinnhverjar","sinnhverjir","sinnhverju","sinnhverjum","sinnhvern","sinnhverra","sinnhverrar","sinnhverri","sinnhvers","sinnhvert","sinnhvor","sinnhvora","sinnhvorar","sinnhvorir","sinnhvorn","sinnhvorra","sinnhvorrar","sinnhvorri","sinnhvors","sinnhvort","sinnhvoru","sinnhvorum","sinni","síns","sínu","sínum","sitt","sitthvað","sitthver","sitthverja","sitthverjar","sitthverjir","sitthverju","sitthverjum","sitthvern","sitthverra","sitthverrar","sitthverri","sitthvers","sitthvert","sitthvor","sitthvora","sitthvorar","sitthvorir","sitthvorn","sitthvorra","sitthvorrar","sitthvorri","sitthvors","sitthvort","sitthvoru","sitthvorum","sjálf","sjálfa","sjálfan","sjálfar","sjálfir","sjálfra","sjálfrar","sjálfri","sjálfs","sjálft","sjálfu","sjálfum","sjálfur","slík","slíka","slíkan","slíkar","slíkir","slíkra","slíkrar","slíkri","slíks","slíkt","slíku","slíkum","slíkur","söm","sömu","sömum","sú","sum","suma","suman","sumar","sumir","sumra","sumrar","sumri","sums","sumt","sumu","sumum","sumur","vettugi","vor","vora","vorar","vorir","vorn","vorra","vorrar","vorri","vors","vort","voru","vorum","ýmis","ýmiss","ýmissa","ýmissar","ýmissi","ýmist","ýmsa","ýmsan","ýmsar","ýmsir","ýmsu","ýmsum","þá","það","þær","þann","þau","þeim","þeir","þeirra","þeirrar","þeirri","þennan","þess","þessa","þessar","þessara","þessarar","þessari","þessi","þessir","þessu","þessum","þetta","þín","þína","þínar","þínir","þinn","þinna","þinnar","þinni","þíns","þínu","þínum","þitt","þónokkra","þónokkrar","þónokkrir","þónokkru","þónokkrum","þónokkuð","þónokkur","þónokkurn","þónokkurra","þónokkurrar","þónokkurri","þónokkurs","þónokkurt","því","þvílík","þvílíka","þvílíkan","þvílíkar","þvílíkir","þvílíkra","þvílíkrar","þvílíkri","þvílíks","þvílíkt","þvílíku","þvílíkum","þvílíkur","að","af","alltað","andspænis","auk","austan","austanundir","austur","á","án","árla","ásamt","bak","eftir","fjarri","fjær","fram","frá","fyrir","gagnstætt","gagnvart","gegn","gegnt","gegnum","handa","handan","hjá","inn","innan","innanundir","í","jafnframt","jafnhliða","kring","kringum","með","meðal","meður","miðli","milli","millum","mót","móti","nálægt","neðan","niður","norðan","nær","nærri","næst","næstum","of","ofan","ofar","óháð","órafjarri","sakir","samfara","samhliða","samkvæmt","samskipa","samtímis","síðan","síðla","snemma","sunnan","sökum","til","tráss","um","umfram","umhverfis","undan","undir","upp","utan","úr","út","útundan","vegna","vestan","vestur","við","viður","yfir","hið","hin","hina","hinar","hinir","hinn","hinna","hinnar","hinni","hins","hinu","hinum","ég","hana","hann","hans","hennar","henni","honum","hún","mér","mig","mín","okkar","okkur","oss","vér","við","vor","yðar","yður","ykkar","ykkur","þá","það","þær","þau","þeim","þeir","þeirra","þér","þess","þið","þig","þín","þú","því","að","annaðhvort","bæði","eða","eður","ef","eftir","ella","ellegar","en","enda","er","fyrst","heldur","hvenær","hvorki","hvort","meðan","nema","né","nú","nær","og","sem","síðan","svo","til","um","uns","utan","ýmist","þar","þá","þegar","þó","þótt","því","//www.althingi.is/dba-bin/ferill.pl","http"]

class WorldCloudManager:
  def __init__(self, indexName, docType, object):
    print(indexName)
    print(docType)
    print(object)
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
    elif (object.get("group_id") != None):
      self.searchTerms = {"group_id": int(object["group_id"])}
      self.collectionIndexName = "groups_"+object["cluster_id"]
      self.colletionIndexDocType = "group"
      self.collectionIndexSearchId = int(object["group_id"])
    elif (object.get("post_id"!= None)):
      self.searchTerms = {"post_id": int(object["post_id"])}
      self.collectionIndexName = "posts_"+object["cluster_id"]
      self.colletionIndexDocType = "post"
      self.collectionIndexSearchId = int(object["post_id"])
    elif (object.get("policy_game_id") != None):
      self.searchTerms = {"policy_game_id": int(object["policy_game_id"])}
      self.collectionIndexName = "policy_games"
      self.colletionIndexDocType = "policy_game"
      self.collectionIndexSearchId = int(object["policy_game_id"])

    print(self.collectionIndexSearchId)
    print(self.collectionIndexName)
    print(self.colletionIndexDocType)
    self.collection = es.get(index=self.collectionIndexName, id=self.collectionIndexSearchId)
    #print(res['_source'])

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

  def getWordCloud(self):
    items = self.getAllItemsFromES()
    #print(items['hits']['hits'])
    words = []
    stop_words = []
    if (self.collection['_source']['language']=="en"):
       stop_words = set(stopwords.words('english'))
    elif (self.collection['_source']['language']=="is"):
       stop_words = icelandic_stop_words
    elif (self.collection['_source']['language']=="fr"):
       stop_words = set(stopwords.words('french'))
    elif (self.collection['_source']['language']=="de"):
       stop_words = set(stopwords.words('german'))
    elif (self.collection['_source']['language']=="it"):
       stop_words = set(stopwords.words('italian'))
    elif (self.collection['_source']['language']=="es"):
       stop_words = set(stopwords.words('spanish'))
    elif (self.collection['_source']['language']=="se"):
       stop_words = set(stopwords.words('swedish'))
    elif (self.collection['_source']['language']=="	tk"):
       stop_words = set(stopwords.words('turkish'))
    elif (self.collection['_source']['language']=="tg"):
       stop_words = set(stopwords.words('tajik'))
    elif (self.collection['_source']['language']=="pt"):
       stop_words = set(stopwords.words('portuguese'))
    elif (self.collection['_source']['language']=="no"):
       stop_words = set(stopwords.words('norwegian'))
    elif (self.collection['_source']['language']=="ne"):
       stop_words = set(stopwords.words('nepali'))
    elif (self.collection['_source']['language']=="kk"):
       stop_words = set(stopwords.words('kazakh'))
    elif (self.collection['_source']['language']=="id"):
       stop_words = set(stopwords.words('indonesian'))
    elif (self.collection['_source']['language']=="hu"):
       stop_words = set(stopwords.words('hungarian'))
    elif (self.collection['_source']['language']=="gr"):
       stop_words = set(stopwords.words('greek'))
    elif (self.collection['_source']['language']=="fi"):
       stop_words = set(stopwords.words('finnish'))
    elif (self.collection['_source']['language']=="nl"):
       stop_words = set(stopwords.words('dutch'))
    elif (self.collection['_source']['language']=="da"):
       stop_words = set(stopwords.words('danish'))
    elif (self.collection['_source']['language']=="ac"):
       stop_words = set(stopwords.words('azerbaijani'))
    elif (self.collection['_source']['language']=="ar"):
       stop_words = set(stopwords.words('arabic'))
    for item in items['hits']['hits']:
      content = None
      if (item["_source"].get("content")):
        content = item["_source"].get("content")
      elif (item["_source"].get("description")):
        content = item["_source"].get("description")

      if (content!=None):

        word_tokens = word_tokenize(content)

        for w in word_tokens:
            if w.lower() not in stop_words and len(w)>3:
                words.append(w.lower())

        #print("SCORE: "+item["_source"].get("lemmatizedContent"))
      else:
        print("Warning: did not find any content for item")
    #print(outItemTexts)
    #print(outItemIds)
    freq_dist = FreqDist(words)

    return freq_dist.most_common(100)
