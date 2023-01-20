
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
from joblib import Parallel, delayed
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
        n_jobs = 1
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
        self.n_jobs = n_jobs

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

        self.alpha = self.pi_prior * np.ones(self.n_components).astype(self.dtype, copy=False)

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


    def _sample_bound(
            self,
            shared_correlates,
            X_matrix,
            window_size,
            mutation,
            context,
            locus,
            count,
            trinuc_distributions,
            weighted_phi,
            logweight_matrix,
        ):


        locus_logmu, _, log_normalizer = self._get_locus_distribution(
                X_matrix=X_matrix, 
                window_size=window_size,
                shared_correlates=shared_correlates
            )
            
        flattened_logweight = logweight_matrix[:, context, mutation] + trinuc_distributions[context, locus]
        
        flattened_logweight -= np.log(np.dot(
                self.delta/self.delta.sum(axis = 1, keepdims = True), 
                trinuc_distributions[:, locus]
            ))
        
        flattened_logweight += np.log(window_size[:, locus]) + locus_logmu[:, locus] - log_normalizer
        
        return np.sum(weighted_phi*flattened_logweight)
        


    def _bound(self,*,
            shared_correlates,
            corpus, 
            gamma,
            entropy_sstats,
            weighted_phis, 
            likelihood_scale = 1.):
        
        Elog_gamma = log_dirichlet_expectation(gamma)
        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_rho = log_dirichlet_expectation(self.rho)

        logweight_matrix = Elog_gamma_g[:,None, None] + Elog_delta[:,:,None] + Elog_rho

        elbo = 0
        
        for Elog_gamma_g, sample, weighted_phi in zip(Elog_gamma, corpus, weighted_phis):
            
            elbo += self._sample_bound(
                **sample,
                shared_correlates = shared_correlates,
                weighted_phi = weighted_phi,
                logweight_matrix = logweight_matrix,
            )
        
        elbo += entropy_sstats
        elbo *= likelihood_scale
        
        elbo += np.sum(-1/2 * (np.square(self.beta_mu) + np.square(self.beta_nu)))
        elbo += np.sum(np.log(self.beta_nu))
        
        elbo += sum(dirichlet_bound(self.alpha, gamma, Elog_gamma))
        
        elbo += sum(dirichlet_bound(np.ones(self.n_contexts), self.delta, Elog_delta))
        
        elbo += np.sum(dirichlet_bound(np.ones((1,3)), self.rho, Elog_rho))

        return elbo


    def _prebuild_phi_matrix(self):

        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_rho = log_dirichlet_expectation(self.rho)

        self.phi_matrix_prebuild = np.exp(Elog_delta)[:,:,None] * np.exp(Elog_rho)


    def _get_locus_distribution(self,*,X_matrix, window_size, shared_correlates):

        if shared_correlates:
            try:
                return self._cached_correlates
            except AttributeError:
                pass

        locus_logmu = self.beta_mu.dot(X_matrix) # n_topics, n_loci
        locus_logstd = self.beta_nu.dot(X_matrix) # n_topics, n_loci
        
        log_normalizer = np.log(
            (window_size * np.exp(locus_logmu + 1/2*np.square(locus_logstd)) ).sum(axis = -1, keepdims = True)
        )

        if shared_correlates:
            self._cached_correlates = (locus_logmu, locus_logstd, log_normalizer)

        return locus_logmu, locus_logstd, log_normalizer


    def _clear_locus_cache(self):
        delattr(self, '_cached_correlates')

    
    def _sample_inference(self,
        shared_correlates,
        gamma,
        X_matrix,
        mutation,
        context,
        locus,
        count,
        trinuc_distributions,
        window_size
    ):

        locus_logmu, _, log_normalizer = self._get_locus_distribution(
                X_matrix=X_matrix, 
                window_size=window_size,
                shared_correlates=shared_correlates
            )
        
        beta_sstats = defaultdict(lambda : np.zeros(self.n_components))
        delta_sstats = np.zeros_like(self.delta)
        rho_sstats = np.zeros_like(self.rho)

        flattend_phi = trinuc_distributions[context,locus]*\
                       self.phi_matrix_prebuild[:,context,mutation]*\
                       1/np.dot(
                            self.delta/self.delta.sum(axis = 1, keepdims = True), 
                            trinuc_distributions[:, locus]
                        )*\
                       window_size[:, locus] * np.exp( locus_logmu[:, locus] - log_normalizer )
        
        count_g = np.array(count)[:,None]
        
        for i in range(self.estep_iterations): # inner e-step loop
            
            old_gamma = gamma.copy()
            exp_Elog_gamma = np.exp(log_dirichlet_expectation(old_gamma)[:,None])
            
            gamma_sstats = np.squeeze(
                    exp_Elog_gamma*np.dot(flattend_phi, count_g/np.dot(flattend_phi.T, exp_Elog_gamma))
                ).astype(self.dtype)
            
            gamma = self.alpha + gamma_sstats
            
            if np.abs(gamma - old_gamma).mean() < self.difference_tol:
                logger.debug('E-step converged after {} iterations'.format(i))
                break
        else:
            logger.info('E-step did not converge. If this happens frequently, consider increasing "estep_iterations".')

        exp_Elog_gamma = np.exp(log_dirichlet_expectation(old_gamma)[:,None])

        phi_matrix = np.outer(exp_Elog_gamma, 1/np.dot(flattend_phi.T, exp_Elog_gamma))*flattend_phi
        weighted_phi = phi_matrix * count_g.T

        for _mutation, _context, _locus, ss in zip(mutation, context, locus, weighted_phi.T):
            rho_sstats[:, _context, _mutation]+=ss
            delta_sstats[:, _context]+=ss
            beta_sstats[_locus]+=ss
        
        entropy_sstats = -np.sum(weighted_phi * np.where(phi_matrix > 0, np.log(phi_matrix), 0.))
        # the "phi_matrix" is only needed to calculate the entropy of q(z), so we calculate
        # the entropy here to save memory
                            
        return gamma, rho_sstats, delta_sstats, beta_sstats, entropy_sstats, weighted_phi

    
    def _inference(self,*,
            shared_correlates,
            corpus,
            gamma,
        ):
        
        self._prebuild_phi_matrix()

        if shared_correlates:
            self._get_locus_distribution(
                X_matrix=corpus[0]['X_matrix'], 
                window_size=corpus[0]['window_size'],
                shared_correlates=shared_correlates
            )

        weighted_phis, gammas = [], []
        rho_sstats = np.zeros_like(self.rho)
        entropy_sstats = 0

        if shared_correlates:
            beta_sstats = defaultdict(lambda : np.zeros(self.n_components))
            delta_sstats = np.zeros_like(self.delta)
        else:
            beta_sstats = []
            delta_sstats = []

        for sample_gamma, sample_rho_sstats, sample_delta_sstats, sample_beta_sstats, \
            sample_entropy_sstats, sample_weighted_phi in \
            Parallel(n_jobs = self.n_jobs)(
                delayed(self._sample_inference)(**sample, gamma = gamma_g,)
                for gamma_g, sample in zip(gamma, corpus)
            ):

            gammas.append(sample_gamma)
            rho_sstats += sample_rho_sstats
            entropy_sstats += sample_entropy_sstats
            weighted_phis.append(sample_weighted_phi)

            if shared_correlates: # pile up sstats
                delta_sstats += sample_delta_sstats
                for locus, sstats in sample_beta_sstats.items():
                    beta_sstats[locus] += sstats

            else:
                delta_sstats.append(sample_delta_sstats)
                beta_sstats.append(sample_beta_sstats)


        return np.array(gamma), rho_sstats, delta_sstats, beta_sstats, entropy_sstats, weighted_phis
    
    
    def _fit(self,*,
        shared_correlates,
        corpus,
        ):
        
        self.n_locus_features, self.n_loci = corpus[0]['X_matrix'].shape
        self.trained = False

        assert self.n_locus_features < self.n_loci, \
            'The feature matrix should be of shape (N_features, N_loci) where N_features << N_loci. The matrix you provided appears to be in the wrong orientation.'

        self._init_variables()
        
        gamma = self._init_doc_variables(len(corpus))
        
        self.bounds = []

        try:
            for epoch in range(self.num_epochs):
                
                logger.info(' E-step ...')
                gamma, rho_sstats, delta_sstats, beta_sstats, \
                    entropy_sstats, weighted_phis = self._inference(
                        shared_correlates=shared_correlates,
                        corpus= corpus,
                        gamma = gamma,
                    )

                print(gamma)
                
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
