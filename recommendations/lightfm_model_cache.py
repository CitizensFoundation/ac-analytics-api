import os
import pickle

def def_get_common_filename(cluster_id, extension, temp):
    filename = "recModels/lightFMCluster"+str(cluster_id)+"."+extension
    if temp:
        filename += ".tmp"
    return filename

def get_lightfm_model_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"model",temp)

def get_lightfm_usersmap_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"usersmap",temp)

def get_lightfm_usersfeat_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"usersfeat",temp)

def get_lightfm_itemsmap_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"itemsmap",temp)

def get_lightfm_itemsfeat_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"itemsfeat",temp)

def get_last_modified_at(cluster_id):
    return os.path.getmtime(get_lightfm_model_filename(cluster_id))

class LightFmModelCache(object):

    _models = {}
    _user_id_maps = {}
    _user_features = {}
    _item_id_maps = {}
    _item_features = {}
    _lastFileModifiedAt = {}

    @classmethod
    def load(cls, cluster_id):
        cls._models[cluster_id] = pickle.load(
            open(get_lightfm_model_filename(cluster_id), "rb"))

        cls._user_id_maps[cluster_id] = pickle.load(
            open(get_lightfm_usersmap_filename(cluster_id), "rb"))

        cls._user_features[cluster_id] = pickle.load(
            open(get_lightfm_usersfeat_filename(cluster_id), "rb"))

        cls._item_id_maps[cluster_id] = pickle.load(
            open(get_lightfm_itemsmap_filename(cluster_id), "rb"))

        cls._item_features[cluster_id] = pickle.load(
            open(get_lightfm_itemsfeatfilename(cluster_id), "rb"))

        cls._lastFileModifiedAt = get_last_modified_at(cluster_id)

    @classmethod
    def save(cls, model, usersmap, usersfeat, itemsmap, itemsfeat, cluster_id):
        if not os.path.exists("recModels"):
          os.makedirs("recModels")

        cls._models[cluster_id] = model
        cls._user_id_maps[cluster_id] = usersmap
        cls._user_features[cluster_id] = usersfeat
        cls._item_id_maps[cluster_id] = itemsmap
        cls._item_features[cluster_id] = itemsfeat

        with open(get_lightfm_model_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._models[cluster_id], f)

        with open(get_lightfm_usersmap_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._user_id_maps[cluster_id], f)

        with open(get_lightfm_usersfeat_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._user_features[cluster_id], f)

        with open(get_lightfm_itemsmap_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._item_id_maps[cluster_id], f)

        with open(get_lightfm_itemsfeat_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._item_features[cluster_id], f)

        os.replace(get_lightfm_model_filename(cluster_id, True), get_lightfm_model_filename(cluster_id, False))
        os.replace(get_lightfm_usersmap_filename(cluster_id, True), get_lightfm_usersmap_filename(cluster_id, False))
        os.replace(get_lightfm_usersfeat_filename(cluster_id, True), get_lightfm_usersfeat_filename(cluster_id, False))
        os.replace(get_lightfm_itemsmap_filename(cluster_id, True), get_lightfm_itemsmap_filename(cluster_id, False))
        os.replace(get_lightfm_itemsfeat_filename(cluster_id, True), get_lightfm_itemsfeat_filename(cluster_id, False))

    @classmethod
    def get(cls, cluster_id):
        if cluster_id not in cls._models or (cluster_id in cls._lastFileModifiedAt and cls._lastFileModifiedAt[cluster_id] != get_last_modified_at(cluster_id)):
            LightFmModelCache.load(cluster_id)

        return cls._models[cluster_id], cls._user_id_maps[cluster_id], cls._user_features[cluster_id], cls._item_id_maps[cluster_id], cls._item_features[cluster_id]
