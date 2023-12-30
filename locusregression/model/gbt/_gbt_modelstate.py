from .._model_state import ModelState, CorpusState
from ._hist_gbt import CustomHistGradientBooster
from numpy import array
from functools import partial


'''def _multinomial_r2(*, y_true, raw_prediction, design_matrix, sample_weight = None, n_threads = 1):
        
        loss_kw = dict(
            y_true = y_true,
            design_matrix = design_matrix,
            sample_weight = sample_weight,
            n_threads = n_threads,
        )

        fit_ll = -_multinomial_loss(
            **loss_kw,
            raw_prediction = raw_prediction,
        )'''


def _get_model_fn(design_matrix, X_tild, 
                  tree_learning_rate = 0.1, 
                  max_depth = 5,
                  l2_regularization = 0.0,
                  random_state = None,
                ):
    
    model = CustomHistGradientBooster(
                loss = 'poisson',
                scoring='loss',
                learning_rate=tree_learning_rate,
                max_depth=max_depth, 
                random_state=random_state, 
                warm_start=True,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=3,
                l2_regularization=l2_regularization,
                verbose=False,
            )
    
    model.fit_binning(X_tild)

    return model


class GBTModelState(ModelState):

    def __init__(self,
                 tree_learning_rate = 0.1, 
                 max_trees_per_iter = 100,
                 max_depth = 5,
                 l2_regularization = 0.0,
                 **kw,
                ):
        
        super().__init__(
            get_model_fn = partial(_get_model_fn,
                tree_learning_rate=tree_learning_rate,
                max_depth=max_depth,
                l2_regularization=l2_regularization,
                random_state=kw['random_state'],
            ),
            **kw
        )

        self.max_trees_per_iter = max_trees_per_iter
        self.predict_from = [None]*self.n_components

    
    def update_rate_model(self, sstats, learning_rate):
        
        for k, (X,y, sample_weights, raw_predictions, design_matrix) in enumerate(
            self._get_features(sstats)
        ):
            
            rate_model = self.rate_models[k]

            try:
                n_fit_trees = rate_model.n_iter_
            except AttributeError:
                n_fit_trees = 0

            rate_model.set_params(
                max_iter = n_fit_trees + self.max_trees_per_iter
            ).fit(
                X, 
                y,
                sample_weight = sample_weights,
                design_matrix = design_matrix,
                raw_predictions = raw_predictions.reshape((-1,1)),
                svi_shrinkage = learning_rate,
            )

            self.predict_from[k] = n_fit_trees


class GBTCorpusState(CorpusState):
    
    def update_mutation_rate(self, model_state):

        _, X_tild = model_state.feature_transformer(
            [self.name],[self.X_matrix]
        )

        self._logmu = array([
            model_state.rate_models[k]._raw_predict_from(
                X_tild, 
                self._logmu[k].reshape((-1,1)), 
                from_iteration = model_state.predict_from[k]
            ).ravel()
            for k in range(self.n_components)
        ])

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(self.corpus.exposures)
