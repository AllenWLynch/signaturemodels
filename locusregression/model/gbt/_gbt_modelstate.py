from .._model_state import ModelState, CorpusState
import numpy as np
from ..base import dirichlet_bound
from .mu_tree import optim_tree as optimize_mu, TreeFitError

class GBTModelState(ModelState):

    def __init__(self,
                fix_signatures = None,
                pseudocounts = 10000,
                tree_learning_rate = 0.1,*,
                n_components, 
                random_state, 
                n_features, 
                empirical_bayes,
                genome_trinuc_distribution,
                negative_subsample = None,
                dtype
                ):
        
        assert isinstance(n_components, int) and n_components > 1
        self.n_components = n_components
        self.n_features = n_features
        self.random_state = random_state
        self.empirical_bayes = empirical_bayes
        self.tree_learning_rate = tree_learning_rate
        
        self.delta = self.random_state.gamma(100, 1/100, 
                                               (n_components, self.n_contexts),
                                              ).astype(dtype, copy=False)
        
        self.omega = self.random_state.gamma(100, 1/100,
                                               (n_components, self.n_contexts, 3),
                                              ).astype(dtype, copy = False)
        
        self.mu_trees = [[] for i in range(n_components)]
        self.nu_trees = [[] for i in range(n_components)]

        self.mu_tree_added = [False]*self.n_components
        self.nu_tree_added = [False]*self.n_components


        if not fix_signatures is None:
            self._fix_signatures(fix_signatures,
                                 n_components = n_components,
                                 genome_trinuc_distribution = genome_trinuc_distribution,
                                 pseudocounts = pseudocounts
                                )
        else:
            self.fixed_signatures = [False]*n_components


        self.update_signature_distribution()

    
    def update_mu_tree(self, sstats, learning_rate):
        
        for k in range(self.n_components):

            try:
                self.mu_trees[k].append(
                    optimize_mu(
                        learning_rate=learning_rate*self.tree_learning_rate,
                        beta_sstats = [{ l : v[k] for l,v in stats.items() } for stats in sstats.beta_sstats],
                        logmus = [logmus[k] for logmus in sstats.logmus],
                        lognus = [lognus[k] for lognus in sstats.lognus],
                        X_matrices = sstats.X_matrices,
                        exposures = sstats.exposures,
                    )
                )

                self.mu_tree_added[k] = True

            except TreeFitError:
                pass


    def update_nu_tree(self, sstats, learning_rate):
        pass

        '''for k in range(self.n_components):

            try:

                self.nu_trees[k].append(
                    optimize_nu(
                        learning_rate=learning_rate,
                        beta_sstats = [{ l : v[k] for l,v in stats.items() } for stats in sstats.beta_sstats],
                        logmus = [logmus[k] for logmus in sstats.logmus],
                        lognus = [lognus[k] for lognus in sstats.lognus],
                        X_matrices = sstats.X_matrices,
                        exposures = sstats.exposures,
                    )
                )

                self.nu_tree_added[k] = True

            except TreeFitError:
                pass'''


    def update_state(self, sstats, learning_rate):
        
        for param in ['mu_tree','lambda','rho']:
            self.__getattribute__('update_' + param)(sstats, learning_rate) # call update function

        self.update_signature_distribution() # update pre-calculated pure functions of model state 



    def get_posterior_entropy(self):

        ent = 0

        ent += sum(dirichlet_bound(np.ones(self.n_contexts), self.delta))
        ent += np.sum(dirichlet_bound(np.ones((1,3)), self.omega))

        return ent


class GBTCorpusState(CorpusState):
    
    def __init__(self, corpus, *, pi_prior, n_components, dtype, random_state, subset_sample=1):
        
        super().__init__(corpus, pi_prior=pi_prior, 
                         n_components = n_components, 
                         dtype = dtype, 
                         subset_sample = subset_sample,
                         random_state = random_state
                         )
        
        self._logmu =  self.random_state.normal(0.,0.01, 
                               (self.n_components, self.n_loci) 
                              ).astype(self.dtype, copy=False)

        self._lognu = np.zeros_like(self._logmu)
                      
    

    def update_mutation_rate(self, model_state):

        try:
            self._logmu = self._logmu + np.array([
                model_state.mu_trees[i][-1](self.X_matrix.T).T \
                if model_state.mu_tree_added[i] else np.zeros(self._logmu.shape[1])
                for i in range(self.n_components)
            ])

            '''self._lognu = self._lognu + np.array([
                model_state.nu_trees[i][-1](self.X_matrix.T).T \
                if model_state.nu_tree_added[i] else np.zeros(self._logmu.shape[1])
                for i in range(self.n_components)
            ])'''

            #self._lognu = self._lognu + model_state.nu_trees[-1](self.X_matrix)
        except IndexError:
            pass        
        
        self.mu_tree_added = [False]*self.n_components
        self.nu_tree_added = [False]*self.n_components


        self._logvar = 1/2*(self._lognu)**2

        if self.corpus.shared_exposures:
            self._log_denom = self._calc_log_denom(self.corpus.exposures)
            self.__delattr__('_logvar')

