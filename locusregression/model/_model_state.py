
import numpy as np
from .base import log_dirichlet_expectation
from .optim_lambda import LambdaOptimizer
from .base import update_alpha, update_tau, dirichlet_bound
from ..simulation import SimulatedCorpus, COSMIC_SIGS
from sklearn.linear_model import PoissonRegressor


def _get_linear_model():
    return PoissonRegressor(
        alpha = 0, 
        solver = 'newton-cholesky',
        warm_start = True,
        fit_intercept=True,
    )

class ModelState:

    n_contexts = 32

    def __init__(self,
                fix_signatures = None,
                pseudocounts = 5000,
                negative_subsample = 1000,*,
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
        self.negative_subsample = negative_subsample

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

        self.update_signature_distribution()

        self.rate_models = [
            get_model_fn()
            for _ in range(n_components)
        ]


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
    

    def update_signature_distribution(self):

        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_omega = log_dirichlet_expectation(self.omega)

        self.signature_distribution = np.exp(Elog_delta)[:,:,None] * np.exp(Elog_omega)


    def update_rho(self, sstats, learning_rate):

        new_rho = np.vstack([
            np.expand_dims(1 + sstats.mutation_sstats[k] if not self.fixed_signatures[k] else self.omega[k], axis = 0)
            for k in range(self.n_components)
        ])

        self._svi_update('omega', new_rho, learning_rate)

    
    def update_lambda(self, sstats, learning_rate):
        
        _delta = LambdaOptimizer.optimize(
            delta0 = self.delta,
            trinuc_distributions = sstats.trinuc_distributions,
            context_sstats = sstats.context_sstats,
            locus_sstats = sstats.locus_sstats,
            fixed_deltas=self.fixed_signatures,
        )

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
        
        X_cat = np.hstack(sstats.X_matrices).T # (n_corpuses*n_loci, n_features)
        exposures = np.concatenate(sstats.exposures).ravel()
        
        for k, beta_arr in enumerate(
            self._convert_beta_sstats_to_array(sstats.beta_sstats, len(exposures))
        ):  
            current_lograte_prediction = np.concatenate([logmus[k] for logmus in sstats.logmus])
            #
            # For poisson model with exposures, have to divide the observations by the exposures
            # then also provide the exposures as sample weights
            #
            yield(
                X_cat,
                beta_arr/exposures,
                exposures,
                current_lograte_prediction.ravel(),
            )
        

    def update_rate_model(self, sstats, learning_rate):

        for k, (X,y, sample_weights, _) in enumerate(
            self._get_features(sstats)
        ):
            
            # store the current model state
            try:
                old_coef = self.rate_models[k].coef_.copy()
                old_intercept = self.rate_models[k].intercept_.copy()
            except AttributeError:
                old_coef = np.zeros(X.shape[1])
                old_intercept = 0

            # update the model with the new suffstats
            self.rate_models[k].fit(
                X, 
                y,
                sample_weight = sample_weights,
            )

            # merge the new model state with the old
            self.rate_models[k].coef_ = self._svi_update_fn(
                old_coef, self.rate_models[k].coef_, learning_rate
            )

            self.rate_models[k].intercept_ = self._svi_update_fn(
                old_intercept, self.rate_models[k].intercept_, learning_rate
            )


    def update_tau(self, sstats, learning_rate):
        
        _tau = update_tau(self.beta_mu, self.beta_nu)
        
        self._svi_update('tau', _tau, learning_rate)


    def update_state(self, sstats, learning_rate):
        
        update_params = ['rate_model','lambda','rho']
        
        #if self.empirical_bayes:
        #    update_params.append('tau')
        for param in update_params:
            self.__getattribute__('update_' + param)(sstats, learning_rate) # call update function

        self.update_signature_distribution() # update pre-calculated pure functions of model state 


    def get_posterior_entropy(self):

        ent = 0

        ent += sum(dirichlet_bound(np.ones(self.n_contexts), self.delta))
        ent += np.sum(dirichlet_bound(np.ones((1,3)), self.omega))

        return ent

    '''def get_posterior_entropy(self):

        ent = 0

        ent += np.sum(1/(self.tau * np.sqrt(np.pi * 2)))
        ent +=  np.sum(-1/(2*self.tau[:,None]**2) * (np.square(self.beta_mu) + np.square(self.beta_nu)))
        ent += np.sum(np.log(self.beta_nu))

        ent += sum(dirichlet_bound(np.ones(self.n_contexts), self.delta))
        ent += np.sum(dirichlet_bound(np.ones((1,3)), self.omega))

        return ent'''
        


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

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(self.corpus.exposures)
        
    
    def _get_baseline_prediction(self, n_components, n_loci, dtype):
        return np.zeros((n_components, n_loci), dtype = dtype)/n_loci


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


    def _calc_log_denom(self, exposures):

        return np.log(
                np.sum( exposures*np.exp(self._logmu), axis = -1, keepdims = True)
            ) + np.log(1/self.subset_sample)


    def update_mutation_rate(self, model_state):
        
        self._logmu = np.array([
            np.log(model_state.rate_models[k].predict(self.X_matrix.T).T)
            for k in range(self.n_components)
        ])

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(self.corpus.exposures)

    
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
        

    def log_denom(self, exposures):

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

    def get_posterior_entropy(self, gammas):
        return np.sum(dirichlet_bound(self.alpha, np.array(gammas)))