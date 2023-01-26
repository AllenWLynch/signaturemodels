
import numpy as np
from .base import dirichlet_bound, log_dirichlet_expectation
from collections import defaultdict
from locusregression.corpus import COSMIC_SORT_ORDER, SIGNATURE_STRINGS, MUTATION_PALETTE
from .optim import M_step_delta
from .optim_beta import BetaOptimizer
from .optim_lambda import LambdaOptimizer
import tqdm
import time

from sklearn.base import BaseEstimator
import logging
logger = logging.getLogger('LocusRegressor')
logger.setLevel(logging.INFO)
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle
from functools import partial

class LocusRegressor(BaseEstimator):
    
    n_contexts = 32

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def __init__(self, 
        n_components = 10,
        seed = 0, 
        dtype = np.float32,
        pi_prior = 1.,
        num_epochs = 100, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        quiet = True,
        n_jobs = 1
    ):
        '''
        Params
        ------
        n_components : int > 0, defuault = 10
            Number of signatures to find in corpus
        pi_prior : float > 0, default = 5.,
            Dirichlet prior parameter - commonly denoted *alpha*.
        n_jobs : int > 0, default = 1
            Number of concurrent jobs to run for E-step calculations.

        Returns
        -------
        LocusRegressor : instance of model

        '''
        self.seed = seed
        self.difference_tol = difference_tol
        self.estep_iterations = estep_iterations
        self.num_epochs = num_epochs
        self.dtype = dtype
        self.bound_tol = bound_tol
        self.pi_prior = pi_prior
        self.n_components = n_components
        self.quiet = quiet
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


    def _sample_bound(self,*,
            shared_correlates,
            X_matrix,
            window_size,
            mutation,
            context,
            locus,
            count,
            feature_names,
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
            corpus, 
            gamma,
            entropy_sstats,
            weighted_phis, 
            likelihood_scale = 1.):
        
        Elog_gamma = log_dirichlet_expectation(gamma)
        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_rho = log_dirichlet_expectation(self.rho)

        elbo = 0
        
        for Elog_gamma_g, sample, weighted_phi in zip(Elog_gamma, corpus, weighted_phis):
            
            elbo += self._sample_bound(
                weighted_phi = weighted_phi,
                logweight_matrix = Elog_gamma_g[:,None, None] + Elog_delta[:,:,None] + Elog_rho,
                **sample,
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
                logger.debug('Recalculating locus distribution.')
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
        try:
            self._cached_correlates
        except AttributeError:
            pass
        else:
            delattr(self, '_cached_correlates')

    
    def _sample_inference(self,
        shared_correlates,
        gamma,
        X_matrix,
        mutation,
        context,
        locus,
        count,
        feature_names,
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
                            
        return gamma, rho_sstats, delta_sstats, \
                beta_sstats, entropy_sstats, weighted_phi

    
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
        delta_sstats = np.zeros_like(self.delta)
        entropy_sstats = 0

        if shared_correlates:
            beta_sstats = defaultdict(lambda : np.zeros(self.n_components))
        else:
            beta_sstats = []

        for sample_gamma, sample_rho_sstats, sample_delta_sstats, sample_beta_sstats, \
            sample_entropy_sstats, sample_weighted_phi in \
            Parallel(n_jobs = self.n_jobs)(
                delayed(self._sample_inference)(**sample, gamma = gamma_g)
                for gamma_g, sample in (zip(gamma, corpus) if self.quiet \
                    else tqdm.tqdm(zip(gamma, corpus), desc = 'Inferring local RVs', total = len(corpus))
                )
            ):

            gammas.append(sample_gamma)
            weighted_phis.append(sample_weighted_phi)
            entropy_sstats += sample_entropy_sstats
            rho_sstats += sample_rho_sstats
            delta_sstats += sample_delta_sstats

            if shared_correlates: # pile up sstats
                
                for locus, sstats in sample_beta_sstats.items():
                    beta_sstats[locus] += sstats

            else:
                beta_sstats.append(sample_beta_sstats)

        self._clear_locus_cache()

        return np.array(gammas), rho_sstats, delta_sstats, beta_sstats, entropy_sstats, weighted_phis
        
    

    def _fit(self,corpus):
        
        self.n_locus_features, self.n_loci = corpus[0]['X_matrix'].shape
        
        self.trained = False
        self.n_samples = len(corpus)

        self.shared_correlates = corpus[0]['shared_correlates']
        self.feature_names = corpus[0]['feature_names']
        
        self.genome_trinuc_distribution = np.sum(
            corpus[0]['trinuc_distributions']*corpus[0]['window_size'],
            -1, keepdims= True
        )

        assert self.n_jobs > 0

        if self.n_jobs > 1:
            logger.warn('Fitting model using {} cores.'.format(self.n_jobs))
        
        if self.shared_correlates:
            logger.info('Sharing genomic correlates between samples for faster inference.')

        assert self.n_locus_features < self.n_loci, \
            'The feature matrix should be of shape (N_features, N_loci) where N_features << N_loci. The matrix you provided appears to be in the wrong orientation.'

        self._init_variables()
        
        gamma = self._init_doc_variables(len(corpus))
        
        self.bounds = []

        try:
            for epoch in range(1,self.num_epochs+1):

                start_time = time.time()
                
                logger.debug(' E-step ...')
                gamma, rho_sstats, delta_sstats, beta_sstats, \
                    entropy_sstats, weighted_phis = self._inference(
                        shared_correlates = self.shared_correlates,
                        corpus= corpus,
                        gamma = gamma,
                    )
                
                logger.debug(' M-step ...')

                for k in range(self.n_components):   
                    
                    if self.shared_correlates:
                        window_sizes = [corpus[0]['window_size']]
                        X_matrices = [corpus[0]['X_matrix']]
                        trinuc = [corpus[0]['trinuc_distributions']]
                        update_beta_sstats = [{l : ss[k] for l,ss in beta_sstats.items()}]

                    else:
                        update_beta_sstats = [
                                {l : ss[k] for l,ss in beta_sstats_sample.items()}
                                for beta_sstats_sample in beta_sstats
                            ]
                        
                        window_sizes = map(lambda x : x['window_size'], corpus)
                        X_matrices = map(lambda x : x['X_matrix'], corpus)
                        trinuc = map(lambda x : x['trinuc_distributions'], corpus)

                    self.beta_mu[k], self.beta_nu[k] = BetaOptimizer.optimize(
                        beta_mu0 = self.beta_mu[k],  
                        beta_nu0 = self.beta_nu[k],
                        beta_sstats = update_beta_sstats,
                        window_sizes=window_sizes,
                        X_matrices=X_matrices,
                    )

                    self.delta[k] = LambdaOptimizer.optimize(self.delta[k],
                        trinuc_distributions = trinuc,
                        beta_sstats = update_beta_sstats,
                        delta_sstats = delta_sstats[k],
                    )

                # update rho distribution
                self.rho = rho_sstats + 1.

                logger.debug("Estimating evidence lower bound ...")
                
                self.bounds.append(
                    self._bound(
                        corpus = corpus, 
                        gamma = gamma,
                        entropy_sstats = entropy_sstats,
                        weighted_phis = weighted_phis, 
                    )
                )

                elapsed_time = time.time() - start_time
                improvement = self.bounds[-1] - (self.bounds[-2] if epoch > 1 else self.bounds[-1])

                logger.info('  Epoch {:<3} complete. |  Bound: {:<10.2f}, improvement: {:<10.2f}  |  Elapsed time: {:<3.1f} seconds.'\
                        .format(epoch, self.bounds[-1], improvement, elapsed_time))

                if epoch > 1:
                    if (self.bounds[-1] - self.bounds[-2]) < self.bound_tol:
                        break

            else:
                logger.info('Model did not converge, consider increasing "estep_iterations" or "num_epochs" parameters.')

        except KeyboardInterrupt:
            pass
        else:
            self.trained = True

        return self


    def fit(self, corpus):
        '''
        Learn parameters of model from corpus.

        Parameters
        ----------
        corpus : locusregression.Corpus, list of dicts
            Pre-compiled corpus loaded into memory

        Returns
        -------
        Model with inferred parameters.

        '''
        return self._fit(corpus)


    def predict(self,corpus):
        '''
        For each sample in corpus, predicts the expectation of the signature compositions.

        Parameters
        ----------
        corpus : locusregression.Corpus, list of dicts
            Pre-compiled corpus loaded into memory

        Returns
        -------
        gamma : np.ndarray of shape (n_samples, n_components)
            Compositions over signatures for each sample

        '''

        if not self.trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        n_samples = len(corpus)
        
        gamma = self._inference(
            gamma = self._init_doc_variables(n_samples),
            corpus = corpus,
            shared_correlates = self.shared_correlates,
        )[0]

        E_gamma = gamma/gamma.sum(-1, keepdims = True)
        
        return E_gamma

    
    def score(self, corpus):
        '''
        Computes the evidence lower bound score for the corpus.

        Parameters
        ----------
        corpus : locusregression.Corpus, list of dicts
            Pre-compiled corpus loaded into memory

        Returns
        -------
        elbo : float
            Evidence lower bound objective score, higher is better.

        '''

        if not self.trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')
            
        n_samples = len(corpus)
        
        gamma, _,_,_, entropy_sstats, weighted_phis = \
            self._inference(
                gamma = self._init_doc_variables(n_samples),
                corpus = corpus,
                shared_correlates = self.shared_correlates,
            )

        return self._bound(
                corpus = corpus, 
                gamma = gamma,
                entropy_sstats = entropy_sstats,
                weighted_phis = weighted_phis,
                likelihood_scale=self.n_samples/n_samples
            )

    def get_phi_locus_distribution(self, corpus):
        '''
        Each mutation in the corpus is associated with a posterior distribution over components -
        what is the probability that that mutation was generated by each component of the model.

        For each locus/region, this function return the number of mutations in that locus weighed
        by the probability that those mutations were generated by each component.

        This basically explains, for component k, where are ther mutations that it generated?

        Parameters
        ----------
        corpus : locusregression.Corpus, list of dicts
            Pre-compiled corpus loaded into memory

        Returns
        -------
        np.ndarray of shape (n_components, n_loci), Where the entry for the k^th component and the l^th locus is given by
        the sum over all mutations which fell in that locus times the probability that mutation was generated by the kth component.
        
        '''

        assert corpus[0]['shared_correlates'], 'This function is only available for corpuses with shared genomic correlates.'

        n_samples = len(corpus)

        _, _, _, beta_sstats, _, _ = self._inference(
                gamma = self._init_doc_variables(n_samples),
                corpus = corpus,
                shared_correlates = self.shared_correlates,
            )

        X_matrix = corpus[0]['X_matrix']

        phis = np.zeros((self.n_components, self.n_loci))
        for l, ss in beta_sstats.items():
            phis[:,l] = ss

        return phis


    def regress_phi(self, corpus):
        '''
        Each mutation is associated with a posterior distribution over components and some 
        correlates which describe its genomic locus. 

        This function returns a method to plot the association between components and correlates.
        '''

        def plot_phi_regression(
            ax = None, figsize = None,*,
            feature_name, component,
            phis, X_matrix, feature_names,
            **plot_kwargs
        ):

            if ax is None:
                fig, ax = plt.subplots(1,1,figsize=figsize)

            assert feature_name in feature_names, f'Feature name {feature_name} is not in this corpus.'

            correlate_idx = dict(zip(feature_names, range(len(feature_names))))[feature_name]
            
            ax.scatter(
                X_matrix[correlate_idx, :],
                phis[component, :],
                **plot_kwargs
            )

            ax.set(ylabel = 'Log Phi', xlabel = 'Correlate: ' + feature_name,
                yscale = 'log')

            for s in ['right','top']:
                ax.spines[s].set_visible(False)

            ax.axis('equal')

            return ax

        locus_phis = self._get_phi_locus_distribution(corpus)
        
        X_matrix = corpus[0]['X_matrix']

        return partial(plot_phi_regression, 
                    phis = locus_phis, 
                    X_matrix = X_matrix,
                    feature_names = corpus[0]['feature_names']
                )


    def get_locus_distribution(self, sample, n_samples = 100):
        '''
        For a sample, calculate the psi matrix - For each component, the distribution over loci.

        Parameters
        ----------
        sample : dict, entry from a corpus
            A sample from a corpus

        Returns
        -------
        np.ndarray of shape (n_components, n_loci), where the entry for the k^th component at the l^th locus
            is the probability of sampling that locus given a mutation was generated by component k.

        '''

        X_matrix = corpus[0]['X_matrix']
        
        posterior_samples = np.random.randn(self.n_components, self.n_locus_features, n_samples)*self.beta_nu[:,:,None]\
                 + self.beta_mu[:,:,None]

        psi_unnormalized = np.exp(
            np.transpose(posterior_samples, [2,0,1]).dot(X_matrix) # n_samples, K, F x F,L -> n_samples,K,L
        )

        expected_psi = np.mean(psi_unnormalized/psi_unnormalized.sum(-1, keepdims = True), axis = 0) # K,L

        return expected_psi
        


    def signature(self, component, 
            monte_carlo_draws = 1000,
            return_error = False):
        '''
        Returns the 96-dimensional channel for a component.

        Parameters
        ----------
        component : int > 0
            Which signature to return
        monte_carlo_draws : int > 0, default = 1000
            How many monte_carlo draws to use to approximate the posterior distribution of the signature.
        return_error : boolean, default = False
            Return 95% confidence interval/highest-density interval on posterior.
        
        Returns
        -------
        signature : np.ndarray of shape (96,)
        errors : np.ndarray of shape(96,)
        
        '''

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
        '''
        Plot signature.
        '''

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


    def plot_coefficients(self, component, 
        ax = None, figsize = (3,5), 
        error_std_width = 3,
        dotsize = 5):
        '''
        Plot regression coefficients with 99% confidence interval.
        '''

        assert isinstance(component, int) and 0 <= component < self.n_components

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize= figsize)

        mu, nu = self.beta_mu[component,:], self.beta_nu[component,:]

        ax.scatter(
            x = mu,
            y= self.feature_names, 
            marker='s', s=dotsize, 
            color='black')

        ax.axvline(x=0, linestyle='--', color='black', linewidth=1)

        std_width = error_std_width/2

        ax.hlines(
            xmax= std_width*nu + mu,
            xmin = -std_width*nu + mu,
            y = self.feature_names, 
            color = 'black'
            )

        ax.set(ylabel = 'Feature', xlabel = 'Coefficient',
               title = 'Component ' + str(component)
               )

        for s in ['left','right','top']:
            ax.spines[s].set_visible(False)

        return ax