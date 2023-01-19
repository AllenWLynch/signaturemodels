
import numpy as np
from .base import M_step_alpha, dirichlet_bound, log_dirichlet_expectation
from collections import defaultdict
from ..corpus.featurization import COSMIC_SORT_ORDER, SIGNATURE_STRINGS, MUTATION_PALETTE
from .optim import M_step_delta, M_step_mu_nu
from ..corpus.interface import get_corpus_lists
from sklearn.base import BaseEstimator
import logging
logger = logging.getLogger('LocusRegressor')
logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
import pickle



class LocusRegressor(BaseEstimator):
    
    n_contexts = 32

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def __init__(self, 
        seed = 0, 
        eval_every = 10,
        dtype = np.float32,
        pi_prior = 5,
        num_epochs = 10000, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        n_components = 10,
        quiet = True,
    ):
        
        self.seed = seed
        self.difference_tol = difference_tol
        self.estep_iterations = estep_iterations
        self.num_epochs = num_epochs
        self.dtype = dtype
        self.bound_tol = bound_tol
        self.pi_prior = pi_prior
        self.n_components = n_components
        self.quiet = quiet
        self.eval_every = eval_every


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
        
    def _init_doc_variables(self, n_samples):

        gamma = self.random_state.gamma(100., 1./100., 
                                   (n_samples, self.n_components) # n_genomes by n_components
                                  ).astype(self.dtype, copy=False)
        return gamma
        
    
    def _init_variables(self):
        
        assert isinstance(self.n_components, int) and self.n_components > 1

        self.random_state = np.random.RandomState(self.seed)

        self.alpha = self.pi_prior = np.ones(self.n_components).astype(self.dtype, copy=False)

        self.delta = self.random_state.gamma(100, 1/100, 
                                               (self.n_components, self.n_contexts),
                                              ).astype(self.dtype, copy=False)
        
        self.rho = self.random_state.gamma(100, 1/100,
                                               (self.n_components, self.n_contexts, 3),
                                              ).astype(self.dtype, copy = False)
        
        
        self.beta_mu = self.random_state.normal(0.,0.01, 
                               (self.n_components, self.n_locus_features) 
                              ).astype(self.dtype, copy=False)

        
        self.beta_nu = self.random_state.gamma(2., 0.005, 
                               (self.n_components, self.n_locus_features)
                              ).astype(self.dtype, copy=False)


    def _bound(self,*,
            gamma, 
            X_matrix,
            window_size,
            mutation,
            context,
            locus,
            count,
            entropy_sstats,
            trinuc_distributions,
            weighted_phis, 
            likelihood_scale = 1.):
        
        
        locus_logmu = self.beta_mu.dot(X_matrix) # n_topics, n_loci
        locus_logstd = self.beta_nu.dot(X_matrix) # n_topics, n_loci
        
        log_normalizer = np.log(
            ( window_size * np.exp(locus_logmu + 1/2*np.square(locus_logstd)) ).sum(axis = -1, keepdims = True)
        )
        
        Elog_gamma = log_dirichlet_expectation(gamma)
        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_rho = log_dirichlet_expectation(self.rho)
        
        elbo = 0
        
        for g in range(len(gamma)):
            
            logweight_matrix = Elog_gamma[g,:][:,None, None] + Elog_delta[:,:,None] + Elog_rho
            
            flattened_logweight = logweight_matrix[:, context[g], mutation[g]] + trinuc_distributions[context[g], locus[g]]
            
            flattened_logweight -= np.log(np.dot(
                    self.delta/self.delta.sum(axis = 1, keepdims = True), 
                    trinuc_distributions[:, locus[g]]
                ))
            
            flattened_logweight += np.log(window_size[:, locus[g]]) + locus_logmu[:, locus[g]] - log_normalizer
            
            elbo += np.sum(weighted_phis[g]*flattened_logweight)
        
        elbo += entropy_sstats    
        elbo *= likelihood_scale
        
        elbo += np.sum(-1/2 * (np.square(self.beta_mu) + np.square(self.beta_nu)))
        elbo += np.sum(np.log(self.beta_nu))
        
        elbo += sum(dirichlet_bound(self.alpha, gamma, Elog_gamma))
        
        elbo += sum(dirichlet_bound(np.ones(self.n_contexts), self.delta, Elog_delta))
        
        elbo += np.sum(dirichlet_bound(np.ones((1,3)), self.rho, Elog_rho))

        return elbo
    
    
    def _inference(self,*,
        gamma,
        X_matrix,
        mutation,
        context,
        locus,
        count,
        trinuc_distributions,
        window_size):
        
        # calculate distribution over loci
        locus_logmu = self.beta_mu.dot(X_matrix) # n_topics, n_loci
        locus_logstd = self.beta_nu.dot(X_matrix) # n_topics, n_loci
        
        log_normalizer = np.log(
            (window_size * np.exp(locus_logmu + 1/2*np.square(locus_logstd)) ).sum(axis = -1, keepdims = True)
        )
        
        delta_sstats = np.zeros_like(self.delta)
        rho_sstats = np.zeros_like(self.rho)
        beta_sstats = defaultdict(lambda : np.zeros(self.n_components))
        entropy_sstats = 0
        
        weighted_phis = []
        _it = range(len(mutation))
        
        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_rho = log_dirichlet_expectation(self.rho)
        phi_matrix_prebuild = np.exp(Elog_delta)[:,:,None] * np.exp(Elog_rho)
        
        eps = np.finfo(self.dtype).eps

        for g in _it: # outer data loop, change for stochastic variational infernce
            
            flattend_phi = trinuc_distributions[context[g],locus[g]]*phi_matrix_prebuild[:,context[g],mutation[g]]\
                                /np.dot(
                                    self.delta/self.delta.sum(axis = 1, keepdims = True), 
                                    trinuc_distributions[:, locus[g]]
                                ) # j * (k,j) / ( (k,m)x(m,j) --> (k,j)
            
            flattend_phi *= window_size[:, locus[g]] * np.exp( locus_logmu[:, locus[g]] - log_normalizer )
            
            count_g = np.array(count[g])[:,None]
            
            for i in range(self.estep_iterations): # inner e-step loop
                
                old_gamma = gamma[g].copy()
                exp_Elog_gamma = np.exp(log_dirichlet_expectation(old_gamma)[:,None])
                
                norm_phi = np.dot(flattend_phi.T, exp_Elog_gamma)
                
                gamma_sstats = np.squeeze(
                        exp_Elog_gamma*np.dot(
                                flattend_phi, count_g/norm_phi
                        )
                    ).astype(self.dtype)
                
                gamma[g] = self.alpha + gamma_sstats
                
                if np.abs(gamma[g] - old_gamma).mean() < self.difference_tol:
                    break
            
            exp_Elog_gamma = np.exp(log_dirichlet_expectation(old_gamma)[:,None])
            #phi_matrix_unnormalized = flattend_phi*exp_Elog_gamma
            #phi_matrix = phi_matrix_unnormalized/phi_matrix_unnormalized.sum(0, keepdims = True)
            #weighted_phi = phi_matrix * count_g.T
            norm_phi = np.dot(flattend_phi.T, exp_Elog_gamma)
            
            phi_matrix = np.outer(exp_Elog_gamma, 1/norm_phi)*flattend_phi
            weighted_phi = phi_matrix * count_g.T
            
            for _mutation, _context, _locus, ss in zip(mutation[g], context[g], locus[g], weighted_phi.T):
                rho_sstats[:, _context, _mutation]+=ss
                delta_sstats[:, _context]+=ss
                beta_sstats[_locus]+=ss
            
            weighted_phis.append(weighted_phi)
            entropy_sstats += -np.sum(weighted_phi * np.where(phi_matrix > 0, np.log(phi_matrix), 0.))
                # the "phi_matrix" is only needed to calculate the entropy of q(z), so we calculate
                # the entropy here to save memory
                              
        return gamma, rho_sstats, delta_sstats, beta_sstats, entropy_sstats, weighted_phis
    
    
    def _fit(self,*,
        X_matrix,
        trinuc_distributions,
        window_size,
        mutation,
        context,
        locus,
        count
        ):

        kwargs = dict(
            X_matrix=X_matrix,
            mutation = mutation,
            context = context,
            locus = locus,
            count = count,
            trinuc_distributions = trinuc_distributions,
            window_size = window_size,
        )
        
        self.n_locus_features, self.n_loci = X_matrix.shape
        self.trained = False

        assert self.n_locus_features < self.n_loci, \
            'The feature matrix should be of shape (N_features, N_loci) where N_features << N_loci. The matrix you provided appears to be in the wrong orientation.'

        self._init_variables()
        
        gamma = self._init_doc_variables(len(mutation))
        
        self.bounds = []

        try:
            for epoch in range(self.num_epochs):
                
                logger.info(' E-step ...')
                gamma, rho_sstats, delta_sstats, beta_sstats, \
                    entropy_sstats, weighted_phis = self._inference(
                        gamma = gamma,
                        **kwargs,
                    )
                
                
                logger.info(' M-step delta ...')
                
                for k in range(self.n_components):
                    
                    self.delta[k] = M_step_delta(
                        delta_sstats = delta_sstats[k,:], 
                        beta_sstats = {l : ss[k] for l,ss in beta_sstats.items()},
                        delta = self.delta[k], 
                        trinuc_distributions = trinuc_distributions
                    ).astype(self.dtype)
                
                
                logger.info(' M-step beta ...')
                
                for k in range(self.n_components):
                    
                    self.beta_mu[k], self.beta_nu[k] = M_step_mu_nu(
                        beta_sstats = {l : ss[k] for l,ss in beta_sstats.items()},
                        beta_mu = self.beta_mu[k],  
                        beta_nu = self.beta_nu[k], 
                        window_size = window_size,
                        X_matrix = X_matrix,
                    )
                    
                    self.beta_mu[k] = self.beta_mu[k].astype(self.dtype)
                    self.beta_nu[k] = self.beta_nu[k].astype(self.dtype)
                    
                '''self.alpha = M_step_alpha(
                        gamma = gamma,
                        alpha = self.alpha,  
                        lr = 1.,
                    )'''
                
                self.rho = rho_sstats + 1. # uniform prior for update

                self.bounds.append(
                    self._bound(
                        gamma = gamma, 
                        weighted_phis = weighted_phis,
                        entropy_sstats = entropy_sstats,
                        **kwargs,
                    )
                )

                logger.info(' Bound: {:.2f}'.format(self.bounds[-1]))

                if epoch > 1:
                    if (self.bounds[-1] - self.bounds[-2]) < self.bound_tol:
                        break

            else:
                logger.warning('Model did not converge, consider increasing "estep_iterations" or "num_epochs" parameters.')

        except KeyboardInterrupt:
            pass
        else:
            self.trained = True

        return self

    @get_corpus_lists
    def fit(self, **kwargs):
        return self._fit(**kwargs)

    @get_corpus_lists
    def predict(self,**kwargs):

        n_samples = len(kwargs['mutation'])
        
        gamma = self._inference(
            gamma = self._init_doc_variables(n_samples),
            **kwargs
        )[0]

        E_gamma = gamma/gamma.sum(-1, keepdims = True)
        
        return E_gamma

    @get_corpus_lists
    def score(self,**kwargs):

        n_samples = len(kwargs['mutation'])
        
        gamma, _,_,_, entropy_sstats, weighted_phis = \
            self._inference(
                gamma = self._init_doc_variables(n_samples),
                **kwargs
            )

        return self._bound(
            gamma = gamma, 
            weighted_phis = weighted_phis,
            entropy_sstats = entropy_sstats,
            **kwargs
        )


    def signature(self, component, 
            monte_carlo_draws = 10000,
            return_error = False):

        assert isinstance(component, int) and 0 <= component < self.n_components
        
        eps_posterior = np.concatenate([
            np.expand_dims(np.random.dirichlet(self.rho[component, j], size = monte_carlo_draws).T, 0)
            for j in range(32)
        ])

        lambda_posterior = np.expand_dims(
            np.random.dirichlet(
                self.delta[component], 
                size = monte_carlo_draws
            ).T, 1)

        signature_posterior = (lambda_posterior*eps_posterior).reshape(-1, monte_carlo_draws)
        error = 1.5*signature_posterior.std(-1)

        posterior_dict = dict(zip( SIGNATURE_STRINGS, zip(signature_posterior.mean(-1), error) ))

        mean, error = list(zip(
            *[posterior_dict[mut] for mut in COSMIC_SORT_ORDER]
        ))

        if return_error:
            return mean, error
        else:
            return mean


    def plot_signature(self, component, ax = None, figsize = (6,1.25)):

        mean, error = self.signature(component, return_error=True)
        
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize= figsize)

        ax.bar(
            height = mean,
            x = COSMIC_SORT_ORDER,
            color = MUTATION_PALETTE,
            width = 1,
            edgecolor = 'white',
            linewidth = 1,
            yerr = error,
            error_kw = {'alpha' : 0.5, 'linewidth' : 0.5}
        )

        for s in ['left','right','top']:
            ax.spines[s].set_visible(False)

        ax.set(
            yticks = [], xticks = [],
            title = 'Component ' + str(component)
            )

        return ax


    def plot_coefficients(self, component, feature_names, 
        ax = None, figsize = (3,5), error_std_width = 3,
        dotsize = 5):

        assert len(feature_names) == self.n_locus_features
        assert isinstance(component, int) and 0 <= component < self.n_components

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize= figsize)

        mu, nu = self.beta_mu[component,:], self.beta_nu[component,:]

        ax.scatter(
            x = mu,
            y=feature_names, 
            marker='s', s=dotsize, 
            color='black')

        ax.axvline(x=0, linestyle='--', color='black', linewidth=1)

        std_width = error_std_width/2

        ax.hlines(
            xmax= std_width*nu + mu,
            xmin = -std_width*nu + mu,
            y = feature_names, 
            color = 'black'
            )

        ax.set(ylabel = 'Feature', xlabel = 'Coefficient',
               title = 'Component ' + str(component)
               )

        for s in ['left','right','top']:
            ax.spines[s].set_visible(False)

        return ax
