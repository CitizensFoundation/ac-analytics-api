import os
import pickle

def get_lightfm_model_filename(cluster_id):
    if not os.path.exists("recModels"):
        os.makedirs("recModels")
    return "recModels/lightFMCluster"+str(cluster_id)+".model"


def get_last_modified_at(cluster_id):
    return os.path.getmtime(get_lightfm_model_filename(cluster_id))

class LightFmModelCache(object):

    _models = {}
    _lastFileModifiedAt = {}

    @classmethod
    def load_model(cls, cluster_id):
        cls._models[cluster_id] = pickle.load(
            open(get_lightfm_model_filename(cluster_id), "rb"))
        cls._lastFileModifiedAt = get_last_modified_at(cluster_id)

    @classmethod
    def save_model(cls, model, cluster_id):
        cls._models[cluster_id] = model
        with open(get_lightfm_model_filename(cluster_id), 'wb') as f:
            pickle.dump(cls._models[cluster_id], f)

    @classmethod
    def get_model(cls, cluster_id):
        if cls._models[cluster_id] == None or cls._lastFileModifiedAt[cluster_id] != get_last_modified_at(cluster_id):
            LightFmModelCache.load_model(cluster_id)

        return cls._models[cluster_id]
