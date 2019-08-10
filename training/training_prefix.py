def makeTrainingPrefix(language, index, object):
    filename_prefix = language+"_"+index+"_"
    if (object.get("domain_id")!=None):
      filename_prefix += object["domain_id"]+"_"+"_"
    elif (object["community_id"] != None):
      filename_prefix += "_"+object["community_id"]+"_"+"_"
    elif (object["group_id"]):
      filename_prefix += "_"+"_"+object["group_id"]+"_"
    elif (object["policy_game_id"]):
      filename_prefix += "_"+"_"+"_"+object["policy_game_id"]

    return filename_prefix
