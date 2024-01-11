from .._model_state import ModelState, CorpusState
from ._hist_gbt import CustomHistGradientBooster
from numpy import array
from functools import partial
from sklearn.preprocessing import OrdinalEncoder

def _get_model_fn(*,
                  design_matrix,
                  features,
                  categorical_features,
                  interaction_groups,
                  tree_learning_rate = 0.1, 
                  max_depth = 5,
                  l2_regularization = 0.0,
                  n_iter_no_change = 3,
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
                n_iter_no_change=n_iter_no_change,
                l2_regularization=l2_regularization,
                categorical_features=categorical_features,
                interaction_cst = interaction_groups,
                verbose=False,
            )
    
    model.fit_binning(features)

    return model


class GBTModelState(ModelState):

    def __init__(self,
                 tree_learning_rate = 0.1, 
                 max_trees_per_iter = 100,
                 max_depth = 5,
                 l2_regularization = 0.0,
                 n_iter_no_change = 3,
                 **kw,
                ):
        
        super().__init__(
            get_model_fn = partial(_get_model_fn,
                tree_learning_rate=tree_learning_rate,
                max_depth=max_depth,
                l2_regularization=l2_regularization,
                n_iter_no_change=n_iter_no_change,
                random_state=kw['random_state'],
            ),
            categorical_encoder=OrdinalEncoder(max_categories=254),
            **kw
        )

        self.max_trees_per_iter = max_trees_per_iter
        self.predict_from = [None]*self.n_components

    
    def update_rate_model(self, sstats, corpus_states, learning_rate):
        
        design_matrix = self._get_design_matrix(corpus_states)
        X = self.feature_transformer.transform(corpus_states)

        for k, (y, sample_weights, lograte_prediction) in enumerate(
            self._get_targets(sstats, corpus_states)
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
                raw_predictions = lograte_prediction.reshape((-1,1)),
                svi_shrinkage = learning_rate,
            )

            self.predict_from[k] = n_fit_trees


class GBTCorpusState(CorpusState):
    
    def update_mutation_rate(self, model_state):

        X = model_state.feature_transformer.transform(
                                    {self.name : self}
                                )

        self._logmu = array([
            model_state.rate_models[k]._raw_predict_from(
                X, 
                self._logmu[k].reshape((-1,1)), 
                from_iteration = model_state.predict_from[k]
            ).ravel()
            for k in range(self.n_components)
        ])

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(model_state, self.corpus.exposures)
