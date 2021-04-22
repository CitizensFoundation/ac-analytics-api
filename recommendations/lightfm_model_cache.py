import os
import pickle

def def_get_common_filename(cluster_id, extension, temp):
    prefix_path = ""
    if "AC_ANALYTICS_MODELS_PREFIX_PATH" in os.environ:
      prefix_path = os.getenv('AC_ANALYTICS_MODELS_PREFIX_PATH')
    filename = f"{prefix_path}rec_models/lightFMCluster"+str(cluster_id)+"."+extension

    if temp:
        filename += ".tmp"
    return filename

def get_lightfm_model_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"model",temp)

def get_lightfm_usersmap_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"usersmap",temp)

def get_lightfm_usersfeat_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"usersfeat",temp)

def get_lightfm_usersfeaturemap_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"usersfeatmap",temp)

def get_lightfm_itemsmap_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"itemsmap",temp)

def get_lightfm_itemsfeat_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"itemsfeat",temp)

def get_lightfm_interactions_filename(cluster_id, temp = False):
   return def_get_common_filename(cluster_id,"interactions",temp)

def get_last_modified_at(cluster_id):
    return os.path.getmtime(get_lightfm_model_filename(cluster_id))

class LightFmModelCache(object):

    _models = {}
    _interactions = {}
    _user_id_maps = {}
    _user_features = {}
    _user_feature_maps = {}
    _item_id_maps = {}
    _item_features = {}
    _lastFileModifiedAt = {}

    @classmethod
    def load(cls, cluster_id):
        print("LOADING MODEL CACHE")
        cls._models[cluster_id] = pickle.load(
            open(get_lightfm_model_filename(cluster_id), "rb"))

        cls._interactions[cluster_id] = pickle.load(
            open(get_lightfm_interactions_filename(cluster_id), "rb"))

        cls._user_id_maps[cluster_id] = pickle.load(
            open(get_lightfm_usersmap_filename(cluster_id), "rb"))

        cls._user_feature_maps[cluster_id] = pickle.load(
            open(get_lightfm_usersfeaturemap_filename(cluster_id), "rb"))

        cls._user_features[cluster_id] = pickle.load(
            open(get_lightfm_usersfeat_filename(cluster_id), "rb"))

        cls._item_id_maps[cluster_id] = pickle.load(
            open(get_lightfm_itemsmap_filename(cluster_id), "rb"))

        cls._item_features[cluster_id] = pickle.load(
            open(get_lightfm_itemsfeat_filename(cluster_id), "rb"))

        cls._lastFileModifiedAt[cluster_id] = get_last_modified_at(cluster_id)

    @classmethod
    def save(cls, model, usersmap, usersfeat, itemsmap, itemsfeat, interactions, user_feature_map, cluster_id):
        if not os.path.exists("rec_models"):
          os.makedirs("rec_models")

        cls._models[cluster_id] = model
        cls._interactions[cluster_id] = interactions

        cls._user_id_maps[cluster_id] = usersmap
        cls._user_features[cluster_id] = usersfeat
        cls._user_feature_maps[cluster_id] = user_feature_map

        cls._item_id_maps[cluster_id] = itemsmap
        cls._item_features[cluster_id] = itemsfeat

        with open(get_lightfm_model_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._models[cluster_id], f)

        with open(get_lightfm_interactions_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._interactions[cluster_id], f)

        with open(get_lightfm_usersmap_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._user_id_maps[cluster_id], f)

        with open(get_lightfm_usersfeaturemap_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._user_feature_maps[cluster_id], f)

        with open(get_lightfm_usersfeat_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._user_features[cluster_id], f)

        with open(get_lightfm_itemsmap_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._item_id_maps[cluster_id], f)

        with open(get_lightfm_itemsfeat_filename(cluster_id, True), 'wb') as f:
            pickle.dump(cls._item_features[cluster_id], f)

        os.replace(get_lightfm_model_filename(cluster_id, True), get_lightfm_model_filename(cluster_id, False))
        os.replace(get_lightfm_interactions_filename(cluster_id, True), get_lightfm_interactions_filename(cluster_id, False))

        os.replace(get_lightfm_usersmap_filename(cluster_id, True), get_lightfm_usersmap_filename(cluster_id, False))
        os.replace(get_lightfm_usersfeaturemap_filename(cluster_id, True), get_lightfm_usersfeaturemap_filename(cluster_id, False))
        os.replace(get_lightfm_usersfeat_filename(cluster_id, True), get_lightfm_usersfeat_filename(cluster_id, False))

        os.replace(get_lightfm_itemsmap_filename(cluster_id, True), get_lightfm_itemsmap_filename(cluster_id, False))
        os.replace(get_lightfm_itemsfeat_filename(cluster_id, True), get_lightfm_itemsfeat_filename(cluster_id, False))

    @classmethod
    def get(cls, cluster_id):
        filed_modified = False

        if cluster_id in cls._lastFileModifiedAt:
            filed_modified = cls._lastFileModifiedAt[cluster_id] != get_last_modified_at(cluster_id)

        if (cluster_id not in cls._models) or filed_modified:
            LightFmModelCache.load(cluster_id)

        return cls._models[cluster_id], cls._user_id_maps[cluster_id], cls._user_features[cluster_id], cls._item_id_maps[cluster_id], cls._item_features[cluster_id], cls._interactions[cluster_id], cls._user_feature_maps[cluster_id]
