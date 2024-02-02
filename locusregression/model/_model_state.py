
import numpy as np
from ._dirichlet_update import update_alpha
from ..simulation import SimulatedCorpus, COSMIC_SIGS
from sklearn.linear_model import PoissonRegressor
from scipy.special import logsumexp
from sklearn.preprocessing import OneHotEncoder
from ._feature_transformer import FeatureTransformer


def _get_linear_model(*args, **kw):
    return PoissonRegressor(
        alpha = 0, 
        solver = 'newton-cholesky',
        warm_start = True,
        fit_intercept = False,
    )


class DummyCorpus:

    def __init__(self, corpus):
        self.trinuc_distributions = corpus.trinuc_distributions
        self.shared_correlates = corpus.shared_correlates


class ModelState:

    n_contexts = 32

    def __init__(self,
                fix_signatures = None,
                pseudocounts = 100000,*,
                corpus_states,
                n_components, 
                n_features, 
                n_loci,
                random_state, 
                empirical_bayes,
                genome_trinuc_distribution,
                dtype,
                get_model_fn = _get_linear_model,
                categorical_encoder = OneHotEncoder(sparse_output=False, drop='first'),
                **kw,
            ):
        
        assert isinstance(n_components, int) and n_components >= 1
        self.n_components = n_components
        self.n_features = n_features
        self.n_loci = n_loci
        self.random_state = random_state
        self.empirical_bayes = empirical_bayes

        self.tau = np.ones(self.n_components)

        self.delta = self.random_state.gamma(100, 1/100, 
                                               (n_components, self.n_contexts),
                                              ).astype(dtype, copy=False)
        
        self.omega = self.random_state.gamma(100, 1/100,
                                               (n_components, self.n_contexts, 3),
                                              ).astype(dtype, copy = False)
        
        if not fix_signatures is None:
            self._fix_signatures(fix_signatures,
                                 n_components = n_components,
                                 genome_trinuc_distribution = genome_trinuc_distribution,
                                 pseudocounts = pseudocounts
                                )
        else:
            self.fixed_signatures = [False]*n_components
        
        self.feature_transformer = FeatureTransformer(
                                        categorical_encoder=categorical_encoder,
                                    ).fit(corpus_states)

        self._fit_corpus_encoder(corpus_states)

        design_matrix = self._get_design_matrix(corpus_states)
        X = self.feature_transformer.transform(corpus_states)

        self.rate_models = [
            get_model_fn(
                design_matrix = design_matrix, 
                features = X,
                categorical_features = self.feature_transformer.list_categorical_features(),
                interaction_groups = self.feature_transformer.list_feature_groups(),
            ) 
            for _ in range(n_components)
        ]

        self.n_distributions = design_matrix.shape[1]

        self.context_models = [
            PoissonRegressor(alpha = 0, fit_intercept=True, warm_start=True) #, solver='newton-cholesky')
            for _ in range(n_components)
        ]


    def _fit_corpus_encoder(self, corpus_states):

        corpus_names = list(corpus_states.keys())

        self.corpus_intercept_encoder_ = OneHotEncoder(
                        sparse_output=True,
                        drop = None,
                    ).fit(
                        np.array(corpus_names).reshape((-1,1))
                    )
        

    def _get_design_matrix(self, corpus_states):
        n_loci = next(iter(corpus_states.values())).n_loci
        labels = np.concatenate([[name]*n_loci for name in corpus_states.keys()])
        # One-hot encode the labels
        encoded_labels = self.corpus_intercept_encoder_.transform(
            labels.reshape((-1,1))
        )
        return encoded_labels
    

    def _fix_signatures(self, fix_signatures,*,
                        n_components, 
                        genome_trinuc_distribution, 
                        pseudocounts = 10000):
            
        assert isinstance(fix_signatures, list) and len(fix_signatures) <= n_components, \
                'fix_signatures must be a list of signature names with a most n_components elements'
        
        self.fixed_signatures = [True]*len(fix_signatures) + [False]*(n_components - len(fix_signatures))

        for i, sig in enumerate(fix_signatures):
            
            try:
                COSMIC_SIGS[sig]
            except KeyError:
                raise ValueError(f'Unknown signature {sig}')
            
            sigmatrix = SimulatedCorpus.cosmic_sig_to_matrix(COSMIC_SIGS[sig])
            
            self.omega[i] = sigmatrix * pseudocounts + 1.
            self.delta[i] = sigmatrix.sum(axis = -1) * pseudocounts/genome_trinuc_distribution.reshape(-1) + 1.


    @staticmethod
    def _svi_update_fn(old_value, new_value, learning_rate):
        return (1-learning_rate)*old_value + learning_rate*new_value
    

    def _svi_update(self, param, new_value, learning_rate):
        
        self.__setattr__(
            param, self._svi_update_fn(self.__getattribute__(param), new_value, learning_rate)
        )

        return self.__getattribute__(param)


    def update_rho(self, sstats, corpus_states, learning_rate):
        
        new_rho = np.vstack([
            np.expand_dims(sstats.mutation_sstats[k], axis = 0)
            for k in range(self.n_components)
        ])

        self._svi_update('omega', new_rho, learning_rate)

    
    def _lambda_update(self, k, sstats, corpus_states):

        def _get_context_exposure(corpus_state):
            return corpus_state.trinuc_distributions @ \
                (corpus_state.exposures.ravel() * np.exp(corpus_state.logmu[k]))

        context_exposure = sum(map(_get_context_exposure, corpus_states.values()))
        target = sstats.context_sstats[k]

        intercept = target.sum()/context_exposure.sum()

        sample_weights = context_exposure*intercept
        X = np.diag(np.ones_like(target))

        return np.exp(
            self.context_models[k]\
            .fit(
                X, 
                target/sample_weights,
                sample_weight=sample_weights/sample_weights.mean()
            ).coef_
        )

    
    def update_lambda(self, sstats, corpus_states, learning_rate):
        
        _delta = np.array([
            self._lambda_update(k, sstats, corpus_states)
            for k in range(self.n_components)
        ])

        self._svi_update('delta', _delta, learning_rate)

    
    @staticmethod
    def _convert_beta_sstats_to_array(k, sstats, len):
        
        def statsdict_to_arr(stats, k):
            arr = np.zeros(len)
            for l,v in stats.items():
                arr[l] = v[k]
            return arr
        
        return np.array([
                statsdict_to_arr(corpusstats, k)
                for corpusstats in sstats.beta_sstats
            ])


    def _get_nucleotide_effect(self, k, trinuc_distributions):
        return self.delta[k] @ trinuc_distributions
    

    def _get_targets(self, sstats, corpus_states):
        
        trinuc_matrix = next(iter(corpus_states.values())).trinuc_distributions
        num_corpuses = len(corpus_states); n_bins=trinuc_matrix.shape[1]

        exposures = np.concatenate(
            [corpus_states[name].exposures for name in sstats.corpus_names],
            axis = 0,
        )

        for k in range(self.n_components):

            current_lograte_prediction = np.array(
                [corpus_states[name].logmu[k] for name in sstats.corpus_names]
            ).ravel()

            context_effect = np.repeat(
                self._get_nucleotide_effect(k, trinuc_matrix)[None,:],
                num_corpuses, axis = 0
            )

            target = self._convert_beta_sstats_to_array(k, sstats, n_bins).ravel()
            eta = (exposures * context_effect).ravel()

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

            yield (
                y_tild,
                sample_weights/sample_weights.mean(), # rescale the weights to mean 1 so that the learning rate is comparable across components and over epochs
                current_lograte_prediction
            )


    def update_rate_model(self, sstats, corpus_states, learning_rate):

        design_matrix = self._get_design_matrix(corpus_states)
        X = self.feature_transformer.transform(corpus_states)

        X = np.hstack([X, design_matrix.toarray()])

        for k, (y, sample_weights, lograte_prediction) in enumerate(
            self._get_targets(sstats, corpus_states)
        ):
            
            # store the current model state (ignore the intercept fits)
            try:
                old_coef = self.rate_models[k].coef_.copy()
            except AttributeError:
                old_coef = np.zeros(X.shape[1])            

            # update the model with the new suffstats
            self.rate_models[k].fit(
                X, 
                y,
                sample_weight=sample_weights,
            )

            # merge the new model state with the old
            self.rate_models[k].coef_ = self._svi_update_fn(
                old_coef, 
                self.rate_models[k].coef_, 
                learning_rate
            )

    def update_state(self, sstats, corpus_states, learning_rate):
        
        update_params = ['rate_model','lambda','rho']
        
        for param in update_params:
            self.__getattribute__('update_' + param)(sstats, corpus_states, learning_rate) # call update function

        

class CorpusState(ModelState):
    '''
    Holds corpus-level parameters, like the current mutation rate estimates and 
    corpus-specific priors over signatures
    '''


    def __init__(self, corpus,*,pi_prior,n_components, dtype, random_state,
                 subset_sample = 1):

        self.corpus = corpus
        self.random_state = random_state
        self.n_components = n_components
        self.dtype = dtype
        self.pi_prior = pi_prior
        self.n_features, self.n_loci = corpus.shape
        self.subset_sample = subset_sample
        
        self.n_samples = len(corpus)
        
        self.alpha = np.ones(self.n_components)\
            .astype(self.dtype, copy=False)*self.pi_prior
        
        self._logmu = self._get_baseline_prediction(
            self.n_components, self.n_loci, self.dtype
        )

        
    def _get_baseline_prediction(self, n_components, n_loci, dtype):
        return np.zeros((n_components, n_loci), dtype = dtype)
    

    def clone_corpusstate(self, corpus):
        
        new_state = self.__class__(
            corpus = corpus,
            pi_prior= self.pi_prior,
            n_components=self.n_components,
            dtype = self.dtype,
            random_state = self.random_state,
            subset_sample=self.subset_sample
        )
        new_state.alpha = self.alpha.copy()

        return new_state
    

    def subset_corpusstate(self, corpus, locus_subset):

        newstate = self.__class__(
            corpus = corpus,
            pi_prior= self.pi_prior,
            n_components=self.n_components,
            dtype = self.dtype,
            random_state = self.random_state,
            subset_sample=len(locus_subset)/self.n_loci
        )

        newstate.alpha = self.alpha.copy()
        newstate._logmu = self._logmu[:, locus_subset]
        newstate._log_denom = self._log_denom

        return newstate
    

    def update_log_denom(self, model_state, exposures):
        self._log_denom = self._calc_log_denom(model_state, exposures)


    def _get_mutation_rate_logits(self, model_state, exposures):
        return np.log(model_state.delta @ self.trinuc_distributions) + self._logmu + np.log(exposures)

    
    def get_log_component_effect_rate(self, model_state, exposures):
        '''
        Returns a (Z x C x L) tensor of the log of the component-wise mutation rate effects
        '''
        locus_effects = (self._logmu + np.log(exposures))[:,None,:]
        signature_effects = np.log(model_state.delta)[:,:,None] + np.log(self.trinuc_distributions)[None,:,:]

        return np.nan_to_num(
            locus_effects + signature_effects - self._log_denom[:,None,:],
            nan = -np.inf
        )
    

    def _calc_log_denom(self, model_state, exposures):
        # (KxC) @ (CxL) |-> (KxL)
        logits = self._get_mutation_rate_logits(model_state, exposures)
        return logsumexp(logits, axis = 1, keepdims = True)


    def update_mutation_rate(self, model_state, from_scratch=False):
        
        design_matrix = model_state._get_design_matrix({self.name : self})
        X = model_state.feature_transformer.transform(
                                    {self.name : self}
                                )

        X = np.hstack([X, design_matrix.toarray()])

        self._logmu = np.array([
            np.log(model_state.rate_models[k].predict(X).T)
            for k in range(self.n_components)
        ])

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(model_state, self.corpus.exposures)

        return self

    
    def update_alpha(self, sstats, learning_rate):
        _alpha = update_alpha(self.alpha, sstats.alpha_sstats[self.corpus.name])
        self._svi_update('alpha', _alpha, learning_rate)


    def set_alpha(self, gammas):
        _alpha = update_alpha(self.alpha, gammas)
        self._svi_update('alpha', _alpha, 1)


    def update_gamma(self, sstats, learning_rate):
        _gamma = sstats.gamma_sstats[self.corpus.name]
        self._svi_update('gamma', _gamma, learning_rate)


    @property
    def logmu(self):
        return self._logmu
    
    @property
    def exposures(self):
        assert self.corpus.shared_exposures
        return self.corpus.exposures
        
    
    def get_log_denom(self, exposures):
        if self.corpus.shared_correlates:
            return self._log_denom # already have this calculated
        else:
            return self._calc_log_denom(exposures)
        

    @property
    def trinuc_distributions(self):
        return self.corpus.trinuc_distributions
    
    @property
    def features(self):
        return self.corpus.features
    
    @property
    def name(self):
        return self.corpus.name
        
    @property
    def feature_names(self):
        return self.corpus.feature_names

    def as_dummy(self):
        self.corpus = DummyCorpus(self.corpus)
        return self
    