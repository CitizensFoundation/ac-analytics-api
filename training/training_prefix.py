def makeTrainingPrefix(language, index, object):
    filename_prefix = index+"_"+language+"_"
    if (object.get("domain_id")!=None):
      filename_prefix += "domain_"+object["domain_id"]
    elif (object["community_id"] != None):
      filename_prefix += "community_"+object["community_id"]
    elif (object["group_id"]):
      filename_prefix += "group_"+object["group_id"]
    elif (object["policy_game_id"]):
      filename_prefix += "policy_game_"+object["policy_game_id"]

    return filename_prefix
