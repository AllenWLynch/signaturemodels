from .model import LocusRegressor, logger, _pseudo_r2
from .gbt import GBTRegressor
from .marginal_model import MarginalModel

def load_model(model):
    return LocusRegressor.load(model)