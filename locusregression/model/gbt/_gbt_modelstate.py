from .._model_state import ModelState, CorpusState
from ._hist_gbt import CustomHistGradientBooster


def _get_model_fn(tree_learning_rate):
    return CustomHistGradientBooster(
                loss = 'poisson',
                learning_rate=tree_learning_rate,
                max_iter=1,
                warm_start=True,
                verbose=False,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=3,
            )


class GBTModelState(ModelState):

    def __init__(self, 
                 tree_learning_rate = 0.1, 
                 max_trees_per_iter = 20,
                 **kw):
        
        super().__init__(
            get_model_fn=lambda : _get_model_fn(tree_learning_rate),
            **kw
        )
        self.max_trees_per_iter = max_trees_per_iter

    
    def update_rate_model(self, sstats, learning_rate):
        
        for k, (X, y, sample_weights, raw_predictions) in enumerate(
            self._get_features(sstats)
        ):

            self.rate_models[k].set_params(
                max_iter = self.n_iter_ + self.max_trees_per_iter
            ).fit(
                X, 
                y,
                sample_weight = sample_weights,
                raw_predictions = raw_predictions,
                svi_shrinkage = learning_rate,
            )


class GBTCorpusState(CorpusState):
    pass
