from ..model import LocusRegressor
from ._gbt_modelstate import ModelState CorpusState
import _gbt_sstats

class GBTRegressort(LocusRegressor):

    MODEL_STATE = ModelState
    CORPUS_STATE = CorpusState
    SSTATS = _gbt_sstats
