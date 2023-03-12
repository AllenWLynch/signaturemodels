from .model import LocusRegressor, logger

def load_model(model):
    return LocusRegressor.load(model)