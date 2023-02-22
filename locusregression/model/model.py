
import numpy as np
from .base import dirichlet_bound, log_dirichlet_expectation, \
        M_step_alpha, update_tau, dirichlet_multinomial_logprob
from scipy import stats
import tqdm

from collections import defaultdict
from locusregression.corpus import COSMIC_SORT_ORDER, SIGNATURE_STRINGS, MUTATION_PALETTE

from .optim_beta import BetaOptimizer
from .optim_lambda import LambdaOptimizer
import time
import warnings

#from sklearn.base import BaseEstimator
import logging
logger = logging.getLogger('LocusRegressor')

import matplotlib.pyplot as plt
import pickle
from functools import partial

class LocusRegressor:
    
    n_contexts = 32

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def __init__(self, 
        n_components = 10,
        seed = 2, 
        dtype = np.float32,
        pi_prior = 5.,
        num_epochs = 300, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        quiet = True,
        n_jobs = 1,
        locus_subsample = 0.125,
        kappa = 0.5,
        tau = 1,
        eval_every = 50,
        time_limit = None,
        empirical_bayes = True,
        batch_size = None,
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
        self.locus_subsample = locus_subsample
        self.kappa = kappa
        self.eval_every = eval_every
        self.tau = tau
        self.time_limit = time_limit
        self.empirical_bayes = empirical_bayes
        self.batch_size = batch_size



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

        self.param_tau = np.ones(self.n_components)

        self.delta = self.random_state.gamma(100, 1/100, 
                                               (self.n_components, self.n_contexts),
                                              ).astype(self.dtype, copy=False)
        
        self.omega = self.random_state.gamma(100, 1/100,
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
            logweight_matrix,
            Elog_gamma,
        ):

        locus_logmu, _, log_normalizer = self._get_locus_distribution(
                X_matrix=X_matrix, 
                window_size=window_size,
                shared_correlates=shared_correlates
            )

        def _get_phis(Elog_gamma):
        
            flattend_phi = trinuc_distributions[context,locus]*\
                        self.phi_matrix_prebuild[:,context,mutation]*\
                        1/np.dot(
                                self.delta/self.delta.sum(axis = 1, keepdims = True), 
                                trinuc_distributions[:, locus]
                            )*\
                        window_size[:, locus] * np.exp( locus_logmu[:, locus] - log_normalizer )
            
            count_g = np.array(count)[:,None]
            
            exp_Elog_gamma = np.exp(Elog_gamma[:,None])
            phi_matrix = np.outer(exp_Elog_gamma, 1/np.dot(flattend_phi.T, exp_Elog_gamma))*flattend_phi
            weighted_phi = phi_matrix * count_g.T

            return phi_matrix, weighted_phi


        flattened_logweight = logweight_matrix[:, context, mutation] + trinuc_distributions[context, locus]
        
        flattened_logweight -= np.log(np.dot(
                self.delta/self.delta.sum(axis = 1, keepdims = True), 
                trinuc_distributions[:, locus]
            ))
        
        flattened_logweight += np.log(window_size[:, locus]) + locus_logmu[:, locus] - log_normalizer

        phi_matrix, weighted_phi = _get_phis(Elog_gamma)

        entropy_sstats = -np.sum(weighted_phi * np.where(phi_matrix > 0, np.log(phi_matrix), 0.))
        
        return np.sum(weighted_phi*flattened_logweight) + entropy_sstats



    def _bound(self,*,
            corpus, 
            gamma,
            likelihood_scale = 1.
        ):
        
        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_rho = log_dirichlet_expectation(self.omega)
        Elog_gamma = log_dirichlet_expectation(gamma)

        elbo = 0

        self._prebuild_phi_matrix()
        
        for Elog_gamma_g, sample in zip(Elog_gamma, corpus):

            elbo += self._sample_bound(
                Elog_gamma = Elog_gamma_g,
                logweight_matrix = Elog_gamma_g[:,None, None] + Elog_delta[:,:,None] + Elog_rho,
                **sample, shared_correlates= self.shared_correlates,
            )
        
        elbo *= likelihood_scale
        
        elbo += np.sum(1/(self.param_tau * np.sqrt(np.pi * 2)))
        elbo +=  np.sum(-1/(2*self.param_tau[:,None]**2) * (np.square(self.beta_mu) + np.square(self.beta_nu)))
        elbo += np.sum(np.log(self.beta_nu))
        
        elbo += sum(dirichlet_bound(self.alpha, gamma, Elog_gamma))
        
        elbo += sum(dirichlet_bound(np.ones(self.n_contexts), self.delta, Elog_delta))
        
        elbo += np.sum(dirichlet_bound(np.ones((1,3)), self.omega, Elog_rho))

        return elbo


    def _prebuild_phi_matrix(self):

        Elog_delta = log_dirichlet_expectation(self.delta)
        Elog_rho = log_dirichlet_expectation(self.omega)

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
        likelihood_scale = 1.,*,
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
        rho_sstats = np.zeros_like(self.omega)

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
            
            gamma = self.alpha + gamma_sstats*likelihood_scale
            
            if np.abs(gamma - old_gamma).mean() < self.difference_tol:
                #logger.debug('E-step converged after {} iterations'.format(i))
                break
        else:
            logger.debug('E-step did not converge. If this happens frequently, consider increasing "estep_iterations".')


        exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma)[:,None])
        phi_matrix = np.outer(exp_Elog_gamma, 1/np.dot(flattend_phi.T, exp_Elog_gamma))*flattend_phi*likelihood_scale
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

    
    def _inference(self,
            likelihood_scale = 1.,*,
            shared_correlates,
            corpus,
            gamma,
        ):
        
        self._prebuild_phi_matrix()

        if shared_correlates:
            first_off = next(iter(corpus))

            self._get_locus_distribution(
                X_matrix=first_off['X_matrix'], 
                window_size=first_off['window_size'],
                shared_correlates=shared_correlates
            )

        weighted_phis, gammas = [], []
        rho_sstats = np.zeros_like(self.omega)
        delta_sstats = np.zeros_like(self.delta)
        entropy_sstats = 0

        if shared_correlates:
            beta_sstats = defaultdict(lambda : np.zeros(self.n_components))
        else:
            beta_sstats = []

        for gamma_g, sample in zip(gamma, corpus):

            sample_gamma, sample_rho_sstats, sample_delta_sstats, sample_beta_sstats, \
            sample_entropy_sstats, sample_weighted_phi = \
                    self._sample_inference(**sample, shared_correlates=self.shared_correlates,
                        gamma = gamma_g, likelihood_scale = likelihood_scale,)

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
                


    @staticmethod
    def svi_update(old, new, learning_rate):
        return (1-learning_rate)*old + learning_rate*new


    def _fit(self,corpus, reinit = True):

        if self.batch_size is None:
            self.batch_size = len(corpus)
        else:
            self.batch_size = min(self.batch_size, len(corpus))
            
        if self.locus_subsample == 1 and self.batch_size > len(corpus):
            
            learning_rate = 1
            svi_inference = False
            logger.warn('Using batch VI updates.')

        else:
            svi_inference = True
            
        sample_svi = self.batch_size < len(corpus)
        locus_svi = self.locus_subsample < 1

        logging.debug(f'Sample SVI: {sample_svi}, Locus SVI {locus_svi}')
            
    
        self.n_locus_features, self.n_loci = next(iter(corpus))['X_matrix'].shape
        n_subsample_loci = int(self.n_loci * self.locus_subsample)

        self.trained = False
        self.n_samples = len(corpus)

        likelihood_scale = 1/(self.locus_subsample * self.batch_size/self.n_samples)

        self.shared_correlates = corpus.shared_correlates
        self.feature_names = next(iter(corpus))['feature_names']
        
        assert self.n_jobs > 0

        if self.n_jobs > 1:
            logger.warn('Fitting model using {} cores.'.format(self.n_jobs))
        
        if self.shared_correlates:
            logger.info('Sharing genomic correlates between samples for faster inference.')

        assert self.n_locus_features < self.n_loci, \
            'The feature matrix should be of shape (N_features, N_loci) where N_features << N_loci. The matrix you provided appears to be in the wrong orientation.'

        if not reinit:
            try:
                self.param_tau
            except AttributeError:
                reinit = True

        if reinit:
            self._init_variables()
            self.bounds, self.elapsed_times = [], []
            total_time = 0
            self.epochs_trained = 0
        else:
            total_time = sum(self.elapsed_times)

        gamma = self._init_doc_variables(len(corpus))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                for epoch in range(self.epochs_trained+1, self.num_epochs+1):

                    start_time = time.time()
                    if svi_inference:
                        self._clear_locus_cache()
                        learning_rate = (self.tau + epoch)**(-self.kappa)

                        if sample_svi:
                            update_samples = self.random_state.choice(self.n_samples, size = self.batch_size, replace = False)
                            inner_corpus = corpus.subset_samples(update_samples)
                        else:
                            inner_corpus = corpus
                        
                        if locus_svi:
                            inner_corpus = corpus.subset_loci(
                                self.random_state.choice(self.n_loci, size = n_subsample_loci, replace = False)
                            )

                    else:
                        inner_corpus = corpus

                    
                    #logger.debug(' E-step ...')
                    new_gamma, rho_sstats, delta_sstats, beta_sstats, \
                        _, _ = self._inference(
                            shared_correlates = self.shared_correlates,
                            corpus= inner_corpus,
                            gamma = gamma if not sample_svi else self._init_doc_variables(self.batch_size),
                            likelihood_scale = likelihood_scale,
                        )
                    
                    if not sample_svi:
                        gamma = self.svi_update(gamma, new_gamma, learning_rate)
                    else:
                        gamma[update_samples] = self.svi_update(gamma[update_samples], new_gamma, learning_rate)
                    
                    #logger.debug(' M-step ...')

                    new_beta_mu, new_beta_nu, new_delta = \
                        np.zeros_like(self.beta_mu), np.zeros_like(self.beta_nu), np.zeros_like(self.delta)


                    trinuc = next(iter(inner_corpus))['trinuc_distributions']
                        
                    if self.shared_correlates:

                        first_off = next(iter(inner_corpus))
                        window_sizes = [first_off['window_size']]
                        X_matrices = [first_off['X_matrix']]
                        
                        update_beta_sstats = lambda k : [{l : ss[k] for l,ss in beta_sstats.items()}]

                    else:

                        def reuse_iterator(key):
                            class reuseiterator:
                                def __iter__(self):
                                    for x in inner_corpus:
                                        yield x[key]

                            return reuseiterator()
                        
                        window_sizes = reuse_iterator('window_size') #[x['window_size'] for x in inner_corpus]
                        X_matrices = reuse_iterator('X_matrix') #[x['X_matrix'] for x in inner_corpus]

                        update_beta_sstats = lambda k : [{l : ss[k] for l,ss in beta_sstats_sample.items()} for beta_sstats_sample in beta_sstats]
                        

                    for k in range(self.n_components):   

                        new_beta_mu[k], new_beta_nu[k] = \
                            BetaOptimizer.optimize(
                                beta_mu0 = self.beta_mu[k],  
                                beta_nu0 = self.beta_nu[k],
                                beta_sstats = update_beta_sstats(k),
                                window_sizes=window_sizes,
                                X_matrices=X_matrices,
                                tau = self.param_tau[k]
                            )

                        new_delta[k] = LambdaOptimizer.optimize(self.delta[k],
                            trinuc_distributions = trinuc,
                            beta_sstats = update_beta_sstats(k),
                            delta_sstats = delta_sstats[k],
                        )

                    self.beta_mu = self.svi_update(self.beta_mu, new_beta_mu, learning_rate)
                    self.beta_nu = self.svi_update(self.beta_nu, new_beta_nu, learning_rate)
                    self.delta = self.svi_update(self.delta, new_delta, learning_rate)

                    # update rho distribution
                    self.omega = self.svi_update(self.omega, rho_sstats + 1., learning_rate)

                    if self.empirical_bayes:

                        self.alpha = self.svi_update(
                            self.alpha, 
                            M_step_alpha(self.alpha, new_gamma),
                            learning_rate    
                        )

                        self.param_tau = self.svi_update(
                            self.param_tau, 
                            update_tau(new_beta_mu, new_beta_nu), 
                            learning_rate
                        )

                    #logger.debug("Estimating evidence lower bound ...")

                    elapsed_time = time.time() - start_time
                    self.elapsed_times.append(elapsed_time)
                    total_time += elapsed_time
                    self.epochs_trained+=1

                    if epoch % self.eval_every == 0:
                        
                        self.bounds.append(
                            self._bound(corpus = corpus, gamma = gamma) if not sample_svi else self.score(corpus)
                        )

                        if len(self.bounds) > 1:

                            improvement = self.bounds[-1] - (self.bounds[-2] if epoch > 1 else self.bounds[-1])

                            logger.info('  Epoch {:<3} complete. | Elapsed time: {:<3.1f} seconds. | Bound: {:<10.2f}, improvement: {:<10.2f} '\
                                    .format(epoch, elapsed_time, self.bounds[-1], improvement,))

                            if (self.bounds[-1] - self.bounds[-2]) < self.bound_tol and not svi_inference:
                                break

                    else:

                        logger.info('  Epoch {:<3} complete. | Elapsed time: {:<3.1f} seconds.'.format(epoch, elapsed_time))

                    if not self.time_limit is None and total_time >= self.time_limit:
                        logger.info('Time limit reached, stopping training.')
                        break

                else:
                    logger.info('Model did not converge, consider increasing "estep_iterations" or "num_epochs" parameters.')


            except KeyboardInterrupt:
                pass
        
        self.trained = True

        self._clear_locus_cache()

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


    def partial_fit(self, corpus):
        return self._fit(corpus, reinit=False)


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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self._bound(
                    corpus = corpus, 
                    gamma = gamma,
                    likelihood_scale=self.n_samples/n_samples,
                )


    def calc_signature_sstats(self, corpus):

        #self._weighted_trinuc_dists = self._get_weighted_trinuc_distribution(corpus)

        self._genome_trinuc_distribution = np.sum(
            self._pop_corpus(corpus)['trinuc_distributions']*\
                self._pop_corpus(corpus)['window_size'],
            -1, keepdims= True
        )

        self._genome_trinuc_distribution/=self._genome_trinuc_distribution.sum()


    def get_phi_locus_distribution(self, corpus):
        '''
        Each mutation in the corpus is associated with a posterior distribution over components -
        what is the probability that that mutation was generated by each component of the model.

        For each locus/region, this function return the number of mutations in that locus weighed
        by the probability that those mutations were generated by each component.

        This basically explains, for component k, where are ther mutations that it generated?

        Parameters
        ----------
        corpus : locusregression.Corpus, list of dicts, or one single record from a corpus
            Pre-compiled corpus loaded into memory

        Returns
        -------
        np.ndarray of shape (n_components, n_loci), Where the entry for the k^th component and the l^th locus is given by
        the sum over all mutations which fell in that locus times the probability that mutation was generated by the kth component.
        
        '''

        if isinstance(corpus, dict):
            corpus = [corpus]
        else:
            assert isinstance(corpus, list) and (len(corpus) == 0 or corpus.shared_correlates), \
                'This function is only available for corpuses with shared genomic correlates.'

        n_samples = len(corpus)

        _, _, _, beta_sstats, _, _ = self._inference(
                gamma = self._init_doc_variables(n_samples),
                corpus = corpus,
                shared_correlates = self.shared_correlates,
            )

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

        locus_phis = self.get_phi_locus_distribution(corpus)
        
        X_matrix = next(iter(corpus))['X_matrix']

        return partial(plot_phi_regression, 
                    phis = locus_phis, 
                    X_matrix = X_matrix,
                    feature_names = next(iter(corpus))['feature_names']
                )


    def get_locus_distribution(self, corpus, n_samples = 100):
        '''
        For a sample, calculate the psi matrix - For each component, the distribution over loci.

        Parameters
        ----------
        corpus : dict, entry from a corpus, or a corpus of length one
            A sample from a corpus

        Returns
        -------
        np.ndarray of shape (n_components, n_loci), where the entry for the k^th component at the l^th locus
            is the probability of sampling that locus given a mutation was generated by component k.

        '''

        if isinstance(corpus, dict):
            corpus = [corpus]
        else:
            assert len(corpus) == 1 or corpus.shared_correlates, \
                'This function is only available for corpuses with shared genomic correlates, or for single records from a corpus.'

        X_matrix = next(iter(corpus))['X_matrix']
        
        posterior_samples = np.random.randn(self.n_components, self.n_locus_features, n_samples)*self.beta_nu[:,:,None]\
                 + self.beta_mu[:,:,None]

        psi_unnormalized = np.exp(
            np.transpose(posterior_samples, [2,0,1]).dot(X_matrix) # n_samples, K, F x F,L -> n_samples,K,L
        )

        expected_psi = np.mean(psi_unnormalized/psi_unnormalized.sum(-1, keepdims = True), axis = 0) # K,L

        return expected_psi


    def _pop_corpus(self, corpus):
        return next(iter(corpus))


    def _get_weighted_trinuc_distribution(self, corpus):

        if corpus.shared_correlates:

            psi_matrix = self.get_locus_distribution(self._pop_corpus(corpus)) #K, L
            trinuc_dist = self._pop_corpus(corpus)['trinuc_distributions'] # C, L
            window_size = self._pop_corpus(corpus)['window_size'] # 1, L

            return psi_matrix.dot((trinuc_dist * window_size).T) # K, C

        else:

            trinuc_dists = []

            for sample in corpus:

                psi_matrix = self.get_locus_distribution(sample) #K, L
                trinuc_dist = sample['trinuc_distributions']
                window_size = sample['window_size']

                trinuc_dists.append(psi_matrix.dot((trinuc_dist * window_size).T)) # G,K,L

            return np.mean(trinuc_dists, axis = 0)


    def get_expected_mutation_rate(self, corpus, n_samples = 100):
        '''
        For a sample, calculate the expected mutation rate. First, we infer the mixture of 
        processes that are active in a sample. Then, we calculate the expected marginal probability
        of sampling each locus given those processes.

        Parameters
        ----------
        corpus : dict, entry from a corpus, or a corpus of length one
            A sample from a corpus

        Returns
        -------
        np.ndarray of shape (n_loci, ), The probability of sampling that locus given the processes
            in a sample.

        '''

        if isinstance(corpus, dict):
            corpus = [corpus]
        else:
            assert len(corpus) == 1, \
                'This function is only available for single records from a corpus.'
            
        psi_matrix = self.get_locus_distribution(corpus, n_samples=n_samples) # K, L

        gamma = self.predict(corpus)

        return np.squeeze(np.dot(gamma, psi_matrix))
    

    def signature(self, component, 
            monte_carlo_draws = 1000,
            return_error = False,
            normalization = 'local',
            raw = False):
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
        
        try:
            if normalization == 'local':
                norm_frequencies = self._weighted_trinuc_dists[component][:,None]
            elif normalization == 'global':
                norm_frequencies = self._genome_trinuc_distribution
            else:
                raise ValueError(f'Normalization option {normalization} does not exist.')
        except AttributeError as err:
            raise AttributeError('User must run "model.calc_signature_sstats(corpus)" before extracting signatures from data.') from err


        eps_posterior = np.concatenate([
            np.expand_dims(np.random.dirichlet(self.omega[component, j], size = monte_carlo_draws).T, 0)
            for j in range(32)
        ])

        lambda_posterior = \
            np.random.dirichlet(
                self.delta[component], 
                size = monte_carlo_draws
            ).T # 32 x M

        lambda_posterior = lambda_posterior*norm_frequencies/ \
                np.sum(lambda_posterior*norm_frequencies, axis = 0)

        signature_posterior = (np.expand_dims(lambda_posterior,1)*eps_posterior).reshape(-1, monte_carlo_draws)
                
        signature_posterior = signature_posterior/signature_posterior.sum(0, keepdims = True)

        error = 1.5*signature_posterior.std(-1)
        mean = signature_posterior.mean(-1)

        if raw:
            return mean

        posterior_dict = dict(zip( SIGNATURE_STRINGS, zip(mean, error) ))

        mean, error = list(zip(
            *[posterior_dict[mut] for mut in COSMIC_SORT_ORDER]
        ))

        if return_error:
            return mean, error
        else:
            return mean


    def plot_signature(self, component, ax = None, figsize = (6,1.25), normalization = 'local'):
        '''
        Plot signature.
        '''

        mean, error = self.signature(component, return_error=True, 
            normalization = normalization)
        
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
               title = 'Component ' + str(component),
               )

        for s in ['left','right','top']:
            ax.spines[s].set_visible(False)

        return ax

    
    def _model_conditional_log_prob(self,*,
            beta, _lambda, rho,
            alpha,
            X_matrix,
            mutation,
            context,
            locus,
            count,
            trinuc_distributions,
            window_size
        ):

        mutation_matrix = _lambda[:,:,None] * rho

        p_m_l = trinuc_distributions[context, locus]*mutation_matrix[:, context, mutation] \
                 / np.dot(_lambda, trinuc_distributions[:,locus]) # K,C x C,L -> K,I
        
        mutrates = window_size*np.exp(beta @ X_matrix)

        p_l = mutrates[:, locus]/np.sum(mutrates, axis = -1, keepdims=True) # K,I
        
        p_ml_z = p_l*p_m_l*alpha[:,None]

        q_z = p_ml_z/p_ml_z.sum(axis = 0, keepdims = True)

        z = np.argmax(np.log(q_z.T) + self.random_state.gumbel(size=q_z.T.shape), axis=1)

        log_p_ml_z = (np.log(count) + np.log(p_m_l)[z, np.arange(len(z))] \
            + np.log(p_l)[z, np.arange(len(z))]) # choose a likelihood for each observation

        log_p_z_alpha = dirichlet_multinomial_logprob(z, np.array(alpha))

        return log_p_z_alpha + np.sum(log_p_ml_z) 

    
    def _global_posterior_samples(self, size = 1000):

        for i in range(size):

            beta = self.beta_nu*self.random_state.randn(*self.beta_mu.shape) + self.beta_mu

            _lambda = np.vstack([
                self.random_state.dirichlet(self.delta[k])[None,:]
                for k in range(self.n_components)
            ])

            rho = np.vstack([
                np.expand_dims(np.vstack([
                    self.random_state.dirichlet(self.omega[k, c])[None,:]
                    for c in range(32)
                ]), axis = 0)
                for k in range(self.n_components)
            ])

            yield {'beta' : beta, '_lambda' : _lambda, 'rho' : rho}


    def _log_posterior_importance_weight(self,*, beta, _lambda, rho):

        p_prior = stats.norm(np.zeros_like(self.beta_mu), self.param_tau[:,None]).pdf(beta)
        p_posterior = stats.norm(self.beta_mu, self.beta_nu).pdf(beta)

        return np.log(p_prior).sum() - np.log(p_posterior).sum()

    
    def IS_marginal_likelihood(self, sample, corpus, alpha = None, size = 1000):
        
        if alpha is None:
            alpha = self.alpha

        logp_samples = []
        for posterior in tqdm.tqdm(self._global_posterior_samples(size = size), ncols =100, total = size):
            logp_samples.append(
                #self._log_posterior_importance_weight(**posterior) + \
                self._model_conditional_log_prob(
                    **posterior, **sample, 
                    X_matrix = corpus.X_matrix,
                    trinuc_distributions= corpus.trinuc_distributions,
                    alpha = alpha,
                )
            )

        return logp_samples

