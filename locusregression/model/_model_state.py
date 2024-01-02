
import numpy as np
from .base import update_alpha, update_tau
from ..simulation import SimulatedCorpus, COSMIC_SIGS
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logsumexp

def _get_linear_model(*args):
    return PoissonRegressor(
        alpha = 0, 
        solver = 'newton-cholesky',
        warm_start = True,
        fit_intercept = False,
    )


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
                **kw,
            ):
        
        assert isinstance(n_components, int) and n_components > 1
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
        
        self.corpus_encoder = self._fit_corpus_encoder(corpus_states)

        design_matrix, X_tild = self.feature_transformer(
            corpus_states.keys(), 
            [corpus_state.X_matrix for corpus_state in corpus_states.values()]
        )

        self.rate_models = [get_model_fn(design_matrix, X_tild) for _ in range(n_components)]
        self.n_distributions = design_matrix.shape[1]

        self.context_models = [
            PoissonRegressor(alpha = 0, fit_intercept=True, solver='newton-cholesky', warm_start=True)
            for _ in range(n_components)
        ]


    def feature_transformer(self, corpus_names, X_matrices):
        
        #corpus_names = sstats.corpus_names; X_matrices = sstats.X_matrices
        _, n_loci = X_matrices[0].shape
        # Concatenate the matrices
        concatenated_matrix = np.hstack(X_matrices).T

        labels = np.concatenate([[name]*n_loci for name in corpus_names])
        # One-hot encode the labels
        encoded_labels = self.corpus_encoder.transform(
            labels.reshape((-1,1))
        )

        # Concatenate the encoded labels with the concatenated matrix
        return (encoded_labels, concatenated_matrix)
    

    def _fit_corpus_encoder(self, corpus_states):
    
        corpus_names = list(corpus_states.keys())

        encoder = OneHotEncoder(
                        sparse_output=True,
                        drop = None,
                    ).fit(
                        np.array(corpus_names).reshape((-1,1))
                    )
        
        return encoder


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


    def update_rho(self, sstats, learning_rate):

        new_rho = np.vstack([
            np.expand_dims(1 + sstats.mutation_sstats[k] if not self.fixed_signatures[k] else self.omega[k], axis = 0)
            for k in range(self.n_components)
        ])

        self._svi_update('omega', new_rho, learning_rate)

    
    def _lambda_update(self, k, sstats):

        context_exposure = sum([
            sstats.trinuc_distributions @ (exposure_.ravel() * np.exp(theta_l[k]))
            for theta_l, exposure_ in zip(sstats.logmus, sstats.exposures)
        ])

        y = sstats.context_sstats[k] #+ 1

        X = np.diag(np.ones_like(y))

        return np.exp(
            self.context_models[k]\
            .fit(
                X, 
                y/context_exposure,
                sample_weight = context_exposure
            ).coef_
        )

    
    def update_lambda(self, sstats, learning_rate):
        
        _delta = np.array([
            self._lambda_update(k, sstats)
            for k in range(self.n_components)
        ])

        self._svi_update('delta', _delta, learning_rate)

    
    def _convert_beta_sstats_to_array(self, beta_sstats, len):
        
        def statsdict_to_arr(stats, k):
            arr = np.zeros(len)
            for l,v in stats.items():
                arr[l] = v[k]
            return arr
        
        for k in range(self.n_components):
            yield np.concatenate([
                statsdict_to_arr(corpusstats, k)
                for corpusstats in beta_sstats
            ])


    def _get_features(self, sstats):

        def _get_nucleotide_effect(k, sstats):
            return np.repeat(
                (self.n_contexts * self.delta[k]/self.delta[k].sum()) @ sstats.trinuc_distributions,
                len(sstats.X_matrices)
            )
        
        n_samples = sstats.X_matrices[0].shape[1]

        design_matrix, X_cat = self.feature_transformer(
            sstats.corpus_names, sstats.X_matrices
        )

        exposures = np.concatenate(sstats.exposures).ravel()
        
        for k, beta_arr in enumerate(
            self._convert_beta_sstats_to_array(sstats.beta_sstats, n_samples)
        ):  

            current_lograte_prediction = np.concatenate([logmus[k] for logmus in sstats.logmus])
            #
            # For poisson model with exposures, have to divide the observations by the exposures
            # then also provide the exposures as sample weights
            #
            signature_exposures = exposures.copy() * _get_nucleotide_effect(k, sstats)

            yield(
                X_cat,
                beta_arr/signature_exposures,
                signature_exposures,
                current_lograte_prediction.ravel(),
                design_matrix,
            )
        
        

    def update_rate_model(self, sstats, learning_rate):

        for k, (X,y, sample_weights, _, design_matrix) in enumerate(
            self._get_features(sstats)
        ):
            
            # store the current model state (ignore the intercept fits)
            try:
                old_coef = self.rate_models[k].coef_.copy()
            except AttributeError:
                old_coef = np.zeros(X.shape[1] + design_matrix.shape[1])
            
            # add the design matrix to the features
            # to encode intercept terms
            X = np.hstack([X, design_matrix.toarray()])

            # update the model with the new suffstats
            self.rate_models[k].fit(
                X, 
                y,
                sample_weight = sample_weights,
            )

            # merge the new model state with the old
            self.rate_models[k].coef_ = self._svi_update_fn(
                old_coef, 
                self.rate_models[k].coef_, 
                learning_rate
            )


    def update_tau(self, sstats, learning_rate):
        
        _tau = update_tau(self.beta_mu, self.beta_nu)
        
        self._svi_update('tau', _tau, learning_rate)


    def update_state(self, sstats, learning_rate):
        
        update_params = ['rate_model','lambda','rho']
        
        for param in update_params:
            self.__getattribute__('update_' + param)(sstats, learning_rate) # call update function

        

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

        #if self.corpus.shared_exposures:
        #    self._log_denom = self._calc_log_denom(self.corpus.exposures)
        
    def _get_baseline_prediction(self, n_components, n_loci, dtype):
        return np.zeros((n_components, n_loci), dtype = dtype)


    def subset_corpusstate(self, corpus, locus_subset):

        newstate = CorpusState(
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


    def _calc_log_denom(self, model_state, exposures):
        # (KxC) @ (CxL) |-> (KxL)
        logits = np.log(model_state.delta @ self.trinuc_distributions) + self._logmu + np.log(exposures)

        return logsumexp(logits, axis = 1, keepdims = True)


    def update_mutation_rate(self, model_state):
        
        design_matrix, X_tild = model_state.feature_transformer(
            [self.name],[self.X_matrix]
        )

        X = np.hstack([X_tild, design_matrix.toarray()])

        self._logmu = np.array([
            np.log(model_state.rate_models[k].predict(X).T)
            for k in range(self.n_components)
        ])

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(model_state, self.corpus.exposures)

    
    @property
    def _log_mutation_rate(self):
        return self._logmu - self.log_denom(np.ones_like(self.corpus.exposures))

    
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
    def X_matrix(self):
        return self.corpus.X_matrix
    
    @property
    def name(self):
        return self.corpus.name
        