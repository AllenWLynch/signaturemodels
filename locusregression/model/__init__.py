from .model import LocusRegressor
from .tune import tune_model
from .gbt.model import LocusRegressor as GBTRegressor

def load_model(model):
    return LocusRegressor.load(model)