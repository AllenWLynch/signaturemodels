from .._model_state import ModelState, CorpusState
import numpy as np
from ..base import dirichlet_bound
from .optim_tree import optimize_mu, optimize_nu

class GBTModelState(ModelState):

    def __init__(self,*,
                n_components, 
                random_state, 
                n_features, 
                dtype):
        
        assert isinstance(n_components, int) and n_components > 1
        self.n_components = n_components
        self.n_features = n_features
        self.random_state = random_state

        self.delta = self.random_state.gamma(100, 1/100, 
                                               (n_components, self.n_contexts),
                                              ).astype(dtype, copy=False)
        
        self.omega = self.random_state.gamma(100, 1/100,
                                               (n_components, self.n_contexts, 3),
                                              ).astype(dtype, copy = False)
        
        self.mu_trees, self.nu_trees = [],[]

        self.update_signature_distribution()

    
    def update_mu_tree(self, sstats, learning_rate):
        
        self.mu_trees.append(
            optimize_mu(sstats, learning_rate)
        )

    def update_nu_tree(self, sstats, learning_rate):
        
        self.nu_trees.append(
            optimize_nu(sstats, learning_rate)
        )


    def update_state(self, sstats, learning_rate):
        
        for param in ['mu_tree','nu_tree','lambda','rho','tau']:
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


class CorpusState(CorpusState):
    
    def __init__(self, corpus, *, pi_prior, n_components, dtype, subset_sample=1):
        super().__init__(corpus, pi_prior=pi_prior, 
                         n_components = n_components, 
                         dtype = dtype, 
                         subset_sample = subset_sample
                         )
        
        self._logmu =  self.random_state.normal(0.,0.01, 
                               (self.n_components, self.n_features) 
                              ).astype(self.dtype, copy=False)

        self._lognu = self.random_state.gamma(2, 0.005, 
                               (self.n_components, self.n_features) 
                              ).astype(self.dtype, copy=False)
    

    def update_mutation_rate(self, model_state):
        
        self._logmu = self._logmu + model_state.mu_trees[-1](self.X_matrix)
        self._lognu = self._lognu + model_state.nu_trees[-1](self.X_matrix)

        self._logvar = 1/2*(self._lognu)**2

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(self.corpus.exposures)
            self.__delattr__('_logvar')

