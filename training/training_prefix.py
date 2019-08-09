def makeTrainingPrefix(index, object):
    filename_prefix = index+"_"
    if (object.get("domain_id")!=None):
      filename_prefix = object["domain_id"]+"_"+"_"
    elif (object["community_id"] != None):
      filename_prefix = "_"+object["community_id"]+"_"
    elif (object["group_id"]):
      filename_prefix = "_"+"_"+object["group_id"]
    else:
      filename_prefix = "_"+"_"+"article_"

    return filename_prefix
