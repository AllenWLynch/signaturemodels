from .model import LocusRegressor, logger
from .gbt import GBTRegressor
from .marginal_model import MarginalModel

def load_model(model):
    return LocusRegressor.load(model)