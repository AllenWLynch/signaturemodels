from .model import LocusRegressor

def load_model(model):
    return LocusRegressor.load(model)