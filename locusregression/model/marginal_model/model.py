
import locusregression
import numpy as np
from locusregression.model._dirichlet_update import feldmans_r2
from locusregression.model.gbt._gbt_modelstate import GBTModelState


def _get_targets(model_state, corpus_states):
    
    trinuc_matrix = next(iter(corpus_states.values())).trinuc_distributions
    num_corpuses = len(corpus_states); n_bins=trinuc_matrix.shape[1]

    exposures = np.concatenate(
        [state.exposures for state in corpus_states.values()],
        axis = 0,
    )

    current_lograte_prediction = np.concatenate(
        [np.zeros_like(state.exposures) for state in corpus_states.values()],
        axis = 0,
    )

    context_effect = np.repeat(
        model_state._get_nucleotide_effect(0, trinuc_matrix)[None,:],
        num_corpuses, axis = 0
    )

    target = np.array([
        state.corpus.get_empirical_mutation_rate()
        for state in corpus_states.values()
    ])

    # need to fit an intercept term here to rescale the targets to mean 1.
    # otherwise, the targets could be so small that the model doesn't learn
    # because the change in likelihood is so small
    intercept = target.sum(axis=1, keepdims=True)/(
                    exposures*context_effect*np.exp(current_lograte_prediction)
                ).sum(axis=1, keepdims=True)

    y = target.ravel()
    sample_weights = (exposures * context_effect * intercept).ravel()

    return (
        y/sample_weights,
        sample_weights/sample_weights.mean(), # rescale the weights to mean 1 so that the learning rate is comparable across components and over epochs
        current_lograte_prediction.ravel()
    )



import logging
logger = logging.getLogger('Marginal model')

class MarginalModel:

    def __init__(self, tree_kw={'max_iter' : 1000}):
        self.tree_kw = tree_kw


    def fit(self, corpus):

        self.model_ = locusregression.GBTRegressor(
            n_components=1,
        )
        self.model_._init_model(corpus)
        
        logger.info('Computing marginal signature for corpus ...')
        
        # initialize the marginal corpus signature for the lambdas
        context_counts = np.zeros(32)
        for sample in corpus:
            for context, weight in zip(sample.context, sample.weight):
                context_counts[context]+=weight

        self.model_state = self.model_.model_state
        self.corpus_states = self.model_.corpus_states

        global_trinuc = corpus.trinuc_distributions.sum(axis = 1)
        self.model_state.delta[0] = context_counts/global_trinuc
        #

        logger.info('Piling up mutations ...')
        y, sample_weight, current_prediction = _get_targets(
                                self.model_state, self.corpus_states
                            )

        logger.info('Normalizing genomic features ...')
        design_matrix = self.model_state._get_design_matrix(self.corpus_states)
        X = self.model_state.feature_transformer.transform(self.corpus_states)

        self.marginal_gbt_ = self.model_state.rate_models[0]
        self.marginal_gbt_.set_params(**self.tree_kw)

        logger.info('Fitting model ...')
        self.marginal_gbt_.fit(
            X,
            y,
            sample_weight=sample_weight,
            raw_predictions=current_prediction[:,None],
            design_matrix = design_matrix,
        )

        return self
    
    
    def predict(self, corpus):

        try:
            corpus_state = self.corpus_states[corpus.name]
        except KeyError:
            raise ValueError(f'Corpus {corpus.name} not found in model.')
        
        new_state = corpus_state.clone_corpusstate(corpus)
        new_state.update_mutation_rate(self.model_state, from_scratch = True)

        return new_state.get_log_component_effect_rate(
                    self.model_state, new_state.exposures
                )[0]


    def score(self, corpus):

        y = corpus.get_empirical_mutation_rate()
        y_hat = self.predict(corpus)

        # the null predictor is just proportional to the size of the windows
        y_null = corpus.trinuc_distributions.sum(0)

        return feldmans_r2(y, y_hat, y_null)
    
