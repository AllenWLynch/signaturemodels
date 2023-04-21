from .model import LocusRegressor, logger
from .gbt import GBTRegressor

def load_model(model):
    return LocusRegressor.load(model)