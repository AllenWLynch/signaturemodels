from .model import LocusRegressor
from .tune import tune_model

def load_model(model):
    return LocusRegressor.load(model)