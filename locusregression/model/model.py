
import numpy as np
from .base import dirichlet_bound, log_dirichlet_expectation, dirichlet_multinomial_logprob

from scipy import stats
import tqdm

from locusregression.corpus import COSMIC_SORT_ORDER, SIGNATURE_STRINGS, MUTATION_PALETTE

from ._model_state import ModelState, CorpusState
import time
import warnings

#from sklearn.base import BaseEstimator
import logging
logger = logging.getLogger('LocusRegressor')

import matplotlib.pyplot as plt
import pickle
from functools import partial
import locusregression.model._sstats as _sstats

class LocusRegressor:

    MODEL_STATE = ModelState
    CORPUS_STATE = CorpusState
    SSTATS = _sstats
    
    n_contexts = 32

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def __init__(self, 
        n_components = 10,
        seed = 2, 
        dtype = np.float32,
        pi_prior = 3.,
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
        
        
    def _init_doc_variables(self, n_samples, is_bound = False):

        random_state = self.random_state if not is_bound else self.bound_random_state

        gamma = random_state.gamma(100., 1./100., 
                                   (n_samples, self.n_components) # n_genomes by n_components
                                  ).astype(self.dtype, copy=False)
        return gamma
    

    @staticmethod
    def get_flattened_phi(*,model_state, sample, corpus_state):

        E_delta = model_state.delta/model_state.delta.sum(axis = 1, keepdims = True)
        trinuc_distributions = corpus_state.trinuc_distributions

        flattend_phi = trinuc_distributions[sample.context,sample.locus]*model_state.signature_distribution[:,sample.context,sample.mutation]\
                       /(E_delta @ trinuc_distributions[:, sample.locus])\
                       * sample.exposures[:, sample.locus] * np.exp( corpus_state.logmu[:, sample.locus] - corpus_state.log_denom(sample.exposures) )
        
        return flattend_phi


    def _sample_bound(self,*,
            sample,
            model_state,
            corpus_state,
        ):
    
        sample_sstats = self._sample_inference(
            gamma0 = self._init_doc_variables(1, is_bound=True)[0],
            sample = sample,
            model_state = model_state,
            corpus_state = corpus_state,
        )

        logweight_matrix = log_dirichlet_expectation(sample_sstats.gamma)[:,None, None] + np.log(model_state.signature_distribution)

        flattened_logweight = logweight_matrix[:, sample.context, sample.mutation] \
                + corpus_state.trinuc_distributions[sample.context, sample.locus]
        
        flattened_logweight -= np.log(np.dot(
                model_state.delta/model_state.delta.sum(axis = 1, keepdims = True), 
                corpus_state.trinuc_distributions[:, sample.locus]
            ))
        
        flattened_logweight += np.log(sample.exposures[:, sample.locus]) + corpus_state.logmu[:, sample.locus] - corpus_state.log_denom(sample.exposures)

        weighted_phi = sample_sstats.weighed_phi

        phi = weighted_phi/np.array(sample.count)[None,:]

        entropy_sstats = -np.sum(weighted_phi * np.where(phi > 0, np.log(phi), 0.))
        
        return np.sum(weighted_phi*flattened_logweight) + entropy_sstats, sample_sstats.gamma


    def _bound(self,*,
            corpus, 
            model_state,
            corpus_states,
            likelihood_scale = 1.
        ):

        elbo = 0
        
        corpus_gammas = {
            corp.name : []
            for corp in corpus.corpuses
        }

        for sample in corpus:

            sample_elbo, sample_gamma = self._sample_bound(
                sample = sample,
                model_state = model_state,
                corpus_state = corpus_states[sample.corpus_name]
            )

            elbo += sample_elbo
            corpus_gammas[sample.corpus_name].append(sample_gamma)
        
        elbo *= likelihood_scale
        
        elbo += model_state.get_posterior_entropy()

        for name, corpus_state in corpus_states.items():
            elbo += corpus_state.get_posterior_entropy(corpus_gammas[name])

        return elbo
    

    def _sample_inference(self,
        likelihood_scale = 1.,*,
        gamma0,
        sample,
        model_state,
        corpus_state,
    ):
        
        flattend_phi = self.get_flattened_phi(
                model_state=model_state,
                sample = sample,
                corpus_state=corpus_state
            )
    
        count_g = np.array(sample.count)[:,None]

        gamma = gamma0.copy()
        
        for s in range(self.estep_iterations): # inner e-step loop
            
            old_gamma = gamma.copy()
            exp_Elog_gamma = np.exp(log_dirichlet_expectation(old_gamma)[:,None])
            
            gamma_sstats = np.squeeze(
                    exp_Elog_gamma*np.dot(flattend_phi, count_g/np.dot(flattend_phi.T, exp_Elog_gamma))
                ).astype(self.dtype)
            
            gamma = corpus_state.alpha + gamma_sstats*likelihood_scale
            
            if np.abs(gamma - old_gamma).mean() < self.difference_tol:
                logger.debug(f'E-step converged after {s} iterations.')
                break
        else:
            logger.debug('E-step did not converge. If this happens frequently, consider increasing "estep_iterations".')

        exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma)[:,None])
        phi_matrix = np.outer(exp_Elog_gamma, 1/np.dot(flattend_phi.T, exp_Elog_gamma))*flattend_phi*likelihood_scale
        
        weighted_phi = phi_matrix * count_g.T

        return self.SSTATS.SampleSstats(
            model_state = model_state,
            sample=sample,
            weighted_phi=weighted_phi,
            gamma=gamma,
        )
        

    
    def _inference(self,
            likelihood_scale = 1.,*,
            corpus,
            model_state,
            corpus_states,
            gamma,
        ):
        
        sstat_collections = {
            corp.name : self.SSTATS.CorpusSstats(corp, model_state, corpus_states[corp.name]) 
            for corp in corpus.corpuses
        }

        for gamma_g, sample in zip(gamma, corpus):

            sstat_collections[sample.corpus_name] += \
                    self._sample_inference(
                        sample = sample,
                        gamma0 = gamma_g, 
                        likelihood_scale = likelihood_scale,
                        model_state = model_state,
                        corpus_state = corpus_states[sample.corpus_name]
                    )

        return self.SSTATS.MetaSstats(sstat_collections, self.model_state)
        

    def update_mutation_rates(self):
        for corpusstate in self.corpus_states.values():
            corpusstate.update_mutation_rate(self.model_state)


    def _fit(self,corpus, reinit = True):
        
        self.batch_size = len(corpus) if self.batch_size is None else min(self.batch_size, len(corpus))
        assert 0 < self.locus_subsample <= 1

        locus_svi, batch_svi = self.locus_subsample < 1, self.batch_size < len(corpus)

        learning_rate_fn = lambda t : (self.tau + t)**(-self.kappa) if locus_svi or batch_svi else 1.

        self.n_locus_features, self.n_loci = corpus.shape
        self.n_samples, self.feature_names = len(corpus), corpus.feature_names
        
        n_subsample_loci = int(self.n_loci * self.locus_subsample)
        likelihood_scale = 1/(self.locus_subsample * self.batch_size/self.n_samples)

        self.random_state = np.random.RandomState(self.seed)
        self.bound_random_state = np.random.RandomState(self.seed + 1)
        
        assert self.n_locus_features < self.n_loci, \
            'The feature matrix should be of shape (N_features, N_loci) where N_features << N_loci. The matrix you provided appears to be in the wrong orientation.'
        self.trained = False

        if not reinit:
            try:
                self.model_state
            except AttributeError:
                reinit = True

        if reinit:
            self.model_state = self.MODEL_STATE(
                n_components=self.n_components,
                random_state=self.random_state,
                n_features=self.n_locus_features,
                dtype = np.float32
            )

            self.bounds, self.elapsed_times, self.total_time, self.epochs_trained = \
                [], [], 0, 0
            
            self.corpus_states = {
                corp.name : self.CORPUS_STATE(corp, 
                            pi_prior=self.pi_prior,
                            n_components=self.n_components,
                            dtype = self.dtype, 
                            random_state = self.random_state,
                            subset_sample=1)
                for corp in corpus.corpuses
            }

            self.update_mutation_rates()


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                for epoch in range(self.epochs_trained+1, self.num_epochs+1):

                    start_time = time.time()
                    
                    if batch_svi:
                        inner_corpus = corpus.subset_samples(
                            self.random_state.choice(self.n_samples, size = self.batch_size, replace = False)
                        )
                    else:
                        inner_corpus = corpus
                    
                    if locus_svi:
                        update_loci = self.random_state.choice(self.n_loci, size = n_subsample_loci, replace = False)
                        inner_corpus = inner_corpus.subset_loci(update_loci)

                        inner_corpus_states = {
                            name : state.subset_corpusstate(inner_corpus.get_corpus(name), update_loci)
                            for name, state in self.corpus_states.items()
                        }
                    else:
                        inner_corpus_states = self.corpus_states
                    

                    sstats = self._inference(
                                likelihood_scale = likelihood_scale,
                                corpus = inner_corpus,
                                model_state = self.model_state,
                                corpus_states = inner_corpus_states,
                                gamma = self._init_doc_variables(len(inner_corpus)),
                            )
                    
                    self.model_state.update_state(sstats, learning_rate_fn(epoch))
                    
                    if epoch >= 10:
                        for corpus_state in self.corpus_states.values():
                            corpus_state.update_alpha(sstats, learning_rate_fn(epoch))

                    self.update_mutation_rates()

                    elapsed_time = time.time() - start_time
                    self.elapsed_times.append(elapsed_time)
                    self.total_time += elapsed_time
                    self.epochs_trained+=1

                    if epoch % self.eval_every == 0:
                        
                        self.bounds.append(self._bound(
                            corpus = corpus,
                            model_state = self.model_state,
                            corpus_states = self.corpus_states,
                            likelihood_scale=1,
                        ))

                        if len(self.bounds) > 1:
                            improvement = self.bounds[-1] - (self.bounds[-2] if epoch > 1 else self.bounds[-1])
                            
                            logger.info(f'  Epoch {epoch:<3} complete. | Elapsed time: {elapsed_time:<3.2f} seconds. '
                                        f'| Bound: { self.bounds[-1]:<10.2f}, improvement: {improvement:<10.2f} ')
                            
                    else:
                        logger.info('  Epoch {:<3} complete. | Elapsed time: {:<3.1f} seconds.'.format(epoch, elapsed_time))

                    if not self.time_limit is None and self.total_time >= self.time_limit:
                        logger.info('Time limit reached, stopping training.')
                        break

                else:
                    pass

            except KeyboardInterrupt:
                pass
        
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
        self._fit(corpus)

        self._calc_signature_sstats(corpus)

        return self


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


    def _calc_signature_sstats(self, corpus):

        #self._weighted_trinuc_dists = self._get_weighted_trinuc_distribution(corpus)

        self._genome_trinuc_distribution = np.sum(
            corpus.trinuc_distributions*\
                corpus.exposures,
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
            normalization = 'global',
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
            np.expand_dims(np.random.dirichlet(self.model_state.omega[component, j], size = monte_carlo_draws).T, 0)
            for j in range(32)
        ])

        lambda_posterior = \
            np.random.dirichlet(
                self.model_state.delta[component], 
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

        mu, nu = self.model_state.beta_mu[component,:], self.model_state.beta_nu[component,:]

        ax.scatter(
            x = mu,
            y= self.feature_names, 
            marker='s', s=dotsize, 
            color='black'
        )

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

