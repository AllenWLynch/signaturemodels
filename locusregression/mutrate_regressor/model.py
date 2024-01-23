
import locusregression
import numpy as np
from locusregression.model._dirichlet_update import feldmans_r2


class MarginalMutationRateModel:

    def __init__(self, tree_kw={'max_iter' : 1000}):
        self.tree_kw = tree_kw


    def fit(self, corpus):

        self.model_ = locusregression.GBTRegressor(
            n_components=1,
        )
        self.model_._init_model(corpus)

        y = corpus.get_empirical_mutation_rate()

        design_matrix = self.model_.model_state._get_design_matrix(self.model_.corpus_states)
        X = self.model_.model_state.feature_transformer.transform(self.model_.corpus_states)

        self.marginal_gbt_ = self.model_.model_state.rate_models[0]
        self.marginal_gbt_.set_params(**self.tree_kw)

        self.marginal_gbt_.fit(
            X,
            y,
            sample_weight=np.ones_like(y),
            raw_predictions=np.zeros_like(y)[:,None],
            design_matrix = design_matrix,
        )

        return self
    
    
    def predict(self, corpus):

        new_corpusstates = self.model_._init_new_corpusstates(corpus)
            
        design_matrix = self.model_.model_state._get_design_matrix(new_corpusstates)
        X = self.model_.model_state.feature_transformer.transform(new_corpusstates)

        theta = self.marginal_gbt_.predict(
            X,
            design_matrix = design_matrix,
        )

        return theta - np.logaddexp.reduce(theta)


    def score(self, corpus):

        y = corpus.get_empirical_mutation_rate()
        y_hat = self.predict(corpus)

        return feldmans_r2(y, y_hat)
    
