import os
import pickle

def def_get_common_filename(cluster_id, extension, temp):
    filename = "recModels/lightFMCluster"+str(cluster_id)+"."+extension
    if temp:
        filename += ".tmp"
    return filename

def get_lightfm_model_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"model",temp)

def get_lightfm_users_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"users",temp)

def get_lightfm_items_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"items",temp)

def get_last_modified_at(cluster_id):
    return os.path.getmtime(get_lightfm_model_filename(cluster_id))

class LightFmModelCache(object):

    _models = {}
    _user_id_maps = {}
    _item_id_maps = {}
    _item_id_mappings = {}
    _lastFileModifiedAt = {}

    @classmethod
    def load_model(cls, cluster_id):
        cls._models[cluster_id] = pickle.load(
            open(get_lightfm_model_filename(cluster_id), "rb"))

        cls._item_id_maps[cluster_id] = pickle.load(
            open(get_lightfm_items_filename(cluster_id), "rb"))

        cls._user_id_maps[cluster_id] = pickle.load(
            open(get_lightfm_users_filename(cluster_id), "rb"))

        cls._lastFileModifiedAt = get_last_modified_at(cluster_id)

    @classmethod
    def save_model(cls, model, users, items, cluster_id):
        if not os.path.exists("recModels"):
          os.makedirs("recModels")

        cls._models[cluster_id] = model
        cls._user_id_maps[cluster_id] = users
        cls._item_id_maps[cluster_id] = items

        with open(get_lightfm_users_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._user_id_maps[cluster_id], f)

        with open(get_lightfm_items_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._item_id_maps[cluster_id], f)

        with open(get_lightfm_model_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._models[cluster_id], f)

        os.replace(get_lightfm_model_filename(cluster_id, True), get_lightfm_model_filename(cluster_id, False))
        os.replace(get_lightfm_users_filename(cluster_id, True), get_lightfm_users_filename(cluster_id, False))
        os.replace(get_lightfm_items_filename(cluster_id, True), get_lightfm_items_filename(cluster_id, False))

    @classmethod
    def get_model(cls, cluster_id):
        if cls._models[cluster_id] == None or cls._lastFileModifiedAt[cluster_id] != get_last_modified_at(cluster_id):
            LightFmModelCache.load_model(cluster_id)

        return cls._models[cluster_id], cls._user_id_maps[cluster_id], cls._item_id_maps[cluster_id]
