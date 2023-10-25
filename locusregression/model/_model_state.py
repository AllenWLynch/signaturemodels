
import numpy as np
from .base import log_dirichlet_expectation
from .optim_beta import BetaOptimizer
from .optim_lambda import LambdaOptimizer
from .base import update_alpha, update_tau, dirichlet_bound

class ModelState:

    n_contexts = 32

    def __init__(self,*,
                n_components, 
                random_state, 
                n_features, 
                dtype):
        
        assert isinstance(n_components, int) and n_components > 1
        self.n_components = n_components
        self.n_features = n_features
        self.random_state = random_state

        self.tau = np.ones(self.n_components)

        self.delta = self.random_state.gamma(100, 1/100, 
                                               (n_components, self.n_contexts),
                                              ).astype(dtype, copy=False)
        
        self.omega = self.random_state.gamma(100, 1/100,
                                               (n_components, self.n_contexts, 3),
                                              ).astype(dtype, copy = False)
        
        
        self.beta_mu = self.random_state.normal(0.,0.01, 
                               (n_components, n_features) 
                              ).astype(dtype, copy=False)

        
        self.beta_nu = self.random_state.gamma(2., 0.005, 
                               (n_components, n_features)
                              ).astype(dtype, copy=False)

        self.update_signature_distribution()

    
    def _svi_update(self, param, new_value, learning_rate):
        
        self.__setattr__(
            param, (1-learning_rate)*self.__getattribute__(param) + learning_rate*new_value
        )

    def update_signature_distribution(self):

        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_omega = log_dirichlet_expectation(self.omega)

        self.signature_distribution = np.exp(Elog_delta)[:,:,None] * np.exp(Elog_omega)


    def update_rho(self, sstats, learning_rate):
        self._svi_update('omega', 1 + sstats.mutation_sstats, learning_rate)

    
    def update_lambda(self, sstats, learning_rate):
        
        _delta = LambdaOptimizer.optimize(
            delta0 = self.delta,
            trinuc_distributions = sstats.trinuc_distributions,
            context_sstats = sstats.context_sstats,
            locus_sstats = sstats.locus_sstats,
        )

        self._svi_update('delta', _delta, learning_rate)


    def update_beta(self, sstats, learning_rate):
        
        beta_mu, beta_nu = BetaOptimizer.optimize(
            beta_mu0 = self.beta_mu, 
            beta_nu0 = self.beta_nu,
            tau = self.tau,
            beta_sstats = sstats.beta_sstats, 
            X_matrices = sstats.X_matrices, 
            exposures = sstats.exposures,
        )

        self._svi_update('beta_mu', beta_mu, learning_rate)
        self._svi_update('beta_nu', beta_nu, learning_rate)


    def update_tau(self, sstats, learning_rate):
        
        _tau = update_tau(self.beta_mu, self.beta_nu)
        
        self._svi_update('tau', _tau, learning_rate)


    def update_state(self, sstats, learning_rate):
        
        for param in ['beta','lambda','rho','tau']:
            self.__getattribute__('update_' + param)(sstats, learning_rate) # call update function

        self.update_signature_distribution() # update pre-calculated pure functions of model state 


    def get_posterior_entropy(self):

        ent = 0

        ent += np.sum(1/(self.tau * np.sqrt(np.pi * 2)))
        ent +=  np.sum(-1/(2*self.tau[:,None]**2) * (np.square(self.beta_mu) + np.square(self.beta_nu)))
        ent += np.sum(np.log(self.beta_nu))

        ent += sum(dirichlet_bound(np.ones(self.n_contexts), self.delta))
        ent += np.sum(dirichlet_bound(np.ones((1,3)), self.omega))

        return ent
        


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

        try:
            newstate._log_denom = self._log_denom
        except AttributeError:
            newstate._logvar = self._logvar[:, locus_subset]

        return newstate


    def _calc_log_denom(self, exposures):

        return np.log(
                np.sum( exposures*np.exp(self._logmu + self._logvar), axis = -1, keepdims = True)
            ) + np.log(1/self.subset_sample)


    def update_mutation_rate(self, model_state):
        
        self._logmu = model_state.beta_mu @ self.X_matrix
        self._logvar = 1/2*(model_state.beta_nu @ self.X_matrix)**2

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(self.corpus.exposures)
            self.__delattr__('_logvar')

    
    def update_alpha(self, sstats, learning_rate):
        
        _alpha = update_alpha(self.alpha, sstats.alpha_sstats[self.corpus.name])
        self._svi_update('alpha', _alpha, learning_rate)


    def update_gamma(self, sstats, learning_rate):
            
        _gamma = sstats.gamma_sstats[self.corpus.name]
        self._svi_update('gamma', _gamma, learning_rate)


    @property
    def logmu(self):
        return self._logmu
    
    @property
    def logvar(self):
        return self._logvar
    

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