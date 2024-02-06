
import locusregression
import numpy as np
#from locusregression.model import _pseudo_r2
from locusregression.model.gbt._gbt_modelstate import GBTModelState


def _get_targets(model_state, corpus_states, target):
        
    trinuc_matrix = next(iter(corpus_states.values())).context_frequencies
    num_corpuses = len(corpus_states)

    exposures = np.concatenate(
        [state.exposures for state in corpus_states.values()],
        axis = 0,
    )

    current_lograte_prediction = np.concatenate(
        [np.zeros_like(state.exposures) for state in corpus_states.values()],
        axis = 0,
    ).ravel()

    context_effect = np.repeat(
        model_state._get_nucleotide_effect(0, trinuc_matrix)[None,:],
        num_corpuses, axis = 0
    )

    eta = (exposures * context_effect).ravel()
    target = target.ravel()

    # rescale the targets to mean 1 so that the learning rate is comparable across components and over epochs
    m = (target/eta).mean()
    sample_weights = eta * m

    # remove any samples with zero weight to avoid divide-by-zero errors
    zero_mask = sample_weights == 0

    if (target[zero_mask] > 0).any():
        raise ValueError('A sample weight is zero but the target is positive')
    else:
        target = target[~zero_mask]
        sample_weights = sample_weights[~zero_mask]
        current_lograte_prediction = current_lograte_prediction[~zero_mask]

    y_tild = target/sample_weights

    return (
        y_tild,
        sample_weights/sample_weights.mean(), # rescale the weights to mean 1 so that the learning rate is comparable across components and over epochs
        current_lograte_prediction
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

        self.model_state = self.model_.model_state
        self.corpus_states = self.model_.corpus_states
        
        logger.info('Computing marginal signature for corpus ...')

        corpus_marginals = [
            state.corpus.get_empirical_mutation_rate()
            for state in self.model_.corpus_states.values()
        ]

        context_counts = np.array(sum(map(lambda x: np.array(x.sum(1)).ravel(), corpus_marginals)))
        
        # initialize the marginal corpus signature for the lambdas
        # vvvvvv
        #context_counts = np.zeros(32)
        #for sample in corpus:
        #    for context, weight in zip(sample.context, sample.weight):
        #        context_counts[context]+=weight

        global_trinuc = corpus.context_frequencies.sum(axis = 1)
        self.model_state.delta[0] = context_counts/global_trinuc
        # ^^^^^^

        mutation_counts = np.array(list(map(
            lambda x : np.array(x.sum(0)).ravel(),
            corpus_marginals
        )))

        logger.info('Piling up mutations ...')
        y, sample_weight, current_prediction = _get_targets(
                                self.model_state, self.corpus_states, mutation_counts
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

        return np.exp(
            new_state.get_log_component_effect_rate(
                self.model_state, new_state.exposures
            )[0]
        )


    def score(self, corpus):

        y = corpus.get_empirical_mutation_rate().toarray()
        y_hat = self.predict(corpus)

        # the null predictor is just proportional to the size of the windows
        y_null = corpus.context_frequencies

        return _pseudo_r2(y, y_hat, y_null)
    
