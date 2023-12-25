
from .base import log_dirichlet_expectation
from locusregression.corpus import COSMIC_SORT_ORDER, SIGNATURE_STRINGS, MUTATION_PALETTE
from locusregression.corpus.sbs_sample import SBSSample
import locusregression.model._sstats as _sstats
import locusregression.model._importance_sampling as IS
from ._model_state import ModelState, CorpusState

# external imports
from joblib import Parallel, delayed
import numpy as np
from scipy.cluster import hierarchy
from scipy.special import logsumexp
import time
import warnings
from functools import partial
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import logging
logger = logging.getLogger(' LocusRegressor')


#@njit(parallel = True)
def _estep_update(exp_Elog_gamma, alpha, flattend_phi, count_g, likelihood_scale = 1):
    gamma_sstats = exp_Elog_gamma*np.dot(flattend_phi, count_g/np.dot(flattend_phi.T, exp_Elog_gamma))
    gamma_sstats = gamma_sstats.reshape(-1)
    return alpha + gamma_sstats*likelihood_scale


def _estep_gamma(
    gamma0,*,
    weight,
    alpha,
    flattened_phi,
    locus_subsample_rate,
    batch_subsample_rate,
    learning_rate,
    dtype,
    estep_iterations = 100,
    difference_tol = 1e-3,
):

    count_g = np.array(weight)[:,None].astype(dtype, copy=False)
    gamma = gamma0.copy()
    
    for s in range(estep_iterations): # inner e-step loop
        
        old_gamma = gamma.copy()
        exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma)[:,None])

        gamma = _estep_update(exp_Elog_gamma, alpha, flattened_phi, count_g,
                                likelihood_scale=1/locus_subsample_rate)

        if np.abs(gamma - old_gamma).mean() < difference_tol:
            break
    else:
        logger.debug('E-step did not converge. If this happens frequently, consider increasing "estep_iterations".')


    gamma = (1 - learning_rate)*gamma0 + learning_rate*gamma

    exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma)[:,None])
    phi_matrix = np.outer(exp_Elog_gamma, 1/np.dot(flattened_phi.T, exp_Elog_gamma))*flattened_phi/(batch_subsample_rate*locus_subsample_rate)
    
    weighted_phi = phi_matrix * count_g.T

    return gamma, weighted_phi




class TimerContext:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        logger.debug(f'{self.name} - time: {elapsed_time:<3.2f}s. ')


class LocusRegressor:

    MODEL_STATE = ModelState
    CORPUS_STATE = CorpusState
    SSTATS = _sstats
    
    n_contexts = 32

    @classmethod
    def sample_params(cls, trial):
        return dict(
            tau = trial.suggest_categorical('tau', [1, 1, 1, 16, 48, 128]),
            kappa = trial.suggest_categorical('kappa', [0.5, 0.5, 0.5, 0.6, 0.7]),
        )

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def __init__(self, 
        n_components = 10,
        seed = 2, 
        dtype = float,
        pi_prior = 3.,
        num_epochs = 300, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        quiet = False,
        n_jobs = 1,
        locus_subsample = 1,
        kappa = 0.5,
        tau = 1,
        eval_every = 50,
        time_limit = None,
        empirical_bayes = False,
        batch_size = None,
        fix_signatures =  None,
        negative_subsample = None,
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
        self.fix_signatures = fix_signatures
        self.negative_subsample = negative_subsample


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        

    @staticmethod
    def get_flattened_phi(*,model_state, sample, corpus_state):
        '''
        Flattened phi is the probability of a mutation and locus for each signature given the current modelstate.
        Shape: N_sigs x N_mutations
        '''

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
            gamma
        ):
    
        sample_sstats, _ = self._sample_inference(
            gamma0 = gamma,
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

        phi = weighted_phi/np.array(sample.weight)[None,:]

        entropy_sstats = -np.sum(weighted_phi * np.where(phi > 0, np.log(phi), 0.))
        
        return np.sum(weighted_phi*flattened_logweight) + entropy_sstats, sample_sstats.gamma


    def _bound(self,*,
            corpus, 
            model_state,
            corpus_states,
            likelihood_scale = 1.,
            gammas = None,
        ):

        elbo = 0
        
        corpus_gammas = {
            corp.name : []
            for corp in corpus.corpuses
        }

        if gammas is None:
            gammas = self._init_doc_variables(len(corpus), is_bound=True)

        for sample, gamma in zip(corpus, gammas):

            sample_elbo, sample_gamma = self._sample_bound(
                sample = sample,
                model_state = model_state,
                corpus_state = corpus_states[sample.corpus_name],
                gamma=gamma,
            )

            elbo += sample_elbo
            corpus_gammas[sample.corpus_name].append(sample_gamma)
        
        elbo *= likelihood_scale
        
        elbo += model_state.get_posterior_entropy()

        for name, corpus_state in corpus_states.items():
            elbo += corpus_state.get_posterior_entropy(corpus_gammas[name])

        return elbo
    

    def _sample_inference(self,
        locus_subsample_rate = 1.,
        batch_subsample_rate = 1.,
        learning_rate = 1.,*,
        gamma0,
        sample,
        model_state,
        corpus_state,
    ):
        
        flattend_phi = self.get_flattened_phi(
                model_state=model_state,
                sample=sample,
                corpus_state=corpus_state
            ).astype(self.dtype, copy=False)

        gamma, weighted_phi = _estep_gamma(
            gamma0 = gamma0,
            weight = sample.weight,
            alpha = corpus_state.alpha,
            flattened_phi = flattend_phi,
            locus_subsample_rate = locus_subsample_rate,
            batch_subsample_rate = batch_subsample_rate,
            learning_rate = learning_rate,
            dtype = self.dtype,
            estep_iterations = self.estep_iterations,
            difference_tol = self.difference_tol,
        )

        return self.SSTATS.SampleSstats(
            model_state = model_state,
            sample=sample,
            weighted_phi=weighted_phi,
            gamma=gamma,
        ), gamma
        
    
    def _inference(self,
            parallel,*,
            locus_subsample_rate,
            batch_subsample_rate,
            learning_rate,
            corpus,
            model_state,
            corpus_states,
            gamma,
        ):

        def generate_estep_fns():
            
            for gamma0, sample in zip(gamma, corpus):

                corpus_state = corpus_states[sample.corpus_name]

                flattend_phi = self.get_flattened_phi(
                    model_state=model_state,
                    sample=sample,
                    corpus_state=corpus_state
                ).astype(self.dtype, copy=False)

                yield partial(
                    _estep_gamma,
                    gamma0 = gamma0,
                    weight = sample.weight,
                    alpha = corpus_state.alpha,
                    flattened_phi = flattend_phi,
                    locus_subsample_rate = locus_subsample_rate,
                    batch_subsample_rate = batch_subsample_rate,
                    learning_rate = learning_rate,
                    dtype = self.dtype,
                    estep_iterations = self.estep_iterations,
                    difference_tol = self.difference_tol,
                )

        gammas=[]

        sstat_collections = {
            corp.name : self.SSTATS.CorpusSstats(corp, model_state, corpus_states[corp.name]) 
            for corp in corpus.corpuses
        }

        for i, (gamma, weighted_phi) in enumerate(parallel(
                delayed(estep_fn)() for estep_fn in generate_estep_fns()
            )):
            
            sstats = self.SSTATS.SampleSstats(
                    model_state = model_state,
                    sample=corpus[i],
                    weighted_phi=weighted_phi,
                    gamma=gamma,
                )

            sstat_collections[corpus[i].corpus_name] += sstats
            gammas.append(gamma)

        return self.SSTATS.MetaSstats(sstat_collections, self.model_state), np.vstack(gammas)
        

    def update_mutation_rates(self):
        for corpusstate in self.corpus_states.values():
            corpusstate.update_mutation_rate(self.model_state)


    def _init_doc_variables(self, n_samples, is_bound = False):

        random_state = self.bound_random_state if is_bound else self.random_state

        gamma = random_state.gamma(100., 1./100., 
                                (n_samples, self.n_components) # n_genomes by n_components
                                ).astype(self.dtype, copy=False)
        return gamma


    def _get_rate_model_parameters(self):
        return {}

    def _init_model(self, corpus):

        self.random_state = np.random.RandomState(self.seed)
        self.bound_random_state = np.random.RandomState(self.seed + 1)

        self.n_locus_features, self.n_loci = corpus.shape
        self.n_samples, self.feature_names = len(corpus), corpus.feature_names

        self._calc_signature_sstats(corpus)

        if not self.fix_signatures is None:
            logger.warn('Fixing signatures: ' + ', '.join(map(str, self.fix_signatures)) + ', these will not be updated during training.')
        

        self.model_state = self.MODEL_STATE(
                n_components=self.n_components,
                random_state=self.random_state,
                n_features=self.n_locus_features,
                n_loci=self.n_loci,
                empirical_bayes = self.empirical_bayes,
                dtype = self.dtype,
                fix_signatures = self.fix_signatures,
                genome_trinuc_distribution = self._genome_trinuc_distribution,
                X_matrices = [
                    corp.X_matrix for corp in corpus.corpuses
                ],
                **self._get_rate_model_parameters()
        )

        
        fix_strs = list(self.fix_signatures) if not self.fix_signatures is None else []
        self.component_names = fix_strs + ['Component ' + str(i) for i in range(self.n_components - len(fix_strs))]

        self.bounds, self.elapsed_times, self.total_time, self.epochs_trained = \
            [], [], 0, 0
        
        self.corpus_states = {
            corp.name : self.CORPUS_STATE(
                            corp, 
                            pi_prior=self.pi_prior,
                            n_components=self.n_components,
                            dtype = self.dtype, 
                            random_state = self.random_state,
                            subset_sample=1
                        )
            for corp in corpus.corpuses
        }

        #self.update_mutation_rates()

        self._gamma = self._init_doc_variables(self.n_samples)


    def _subsample_corpus(self,*,corpus, batch_svi, locus_svi, n_subsample_loci):

        if batch_svi:
            update_samples = np.sort( self.random_state.choice(self.n_samples, size = self.batch_size, replace = False) )
            inner_corpus = corpus.subset_samples(update_samples)
        else:
            update_samples = np.arange(self.n_samples)
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

        return inner_corpus, inner_corpus_states, update_samples



    def _fit(self,corpus, reinit = True):
        
        self.batch_size = len(corpus) if self.batch_size is None else min(self.batch_size, len(corpus))
        assert 0 < self.locus_subsample <= 1

        locus_svi, batch_svi = self.locus_subsample < 1, self.batch_size < len(corpus)

        learning_rate_fn = lambda t : (self.tau + t)**(-self.kappa) if locus_svi or batch_svi else 1.

        if not reinit:
            try:
                self.model_state
            except AttributeError:
                reinit = True

        if reinit:
            logger.info('Initializing model ...')
            self._init_model(corpus)
        
        n_subsample_loci = int(self.n_loci * self.locus_subsample)
        
        assert self.n_locus_features < self.n_loci, \
            'The feature matrix should be of shape (N_features, N_loci) where N_features << N_loci. The matrix you provided appears to be in the wrong orientation.'
        self.trained = False

        
        with warnings.catch_warnings(), \
            Parallel(n_jobs=self.n_jobs, return_as = 'generator', batch_size = 10) as parallel:            
            warnings.simplefilter("ignore")

            logger.info('Training model ...')

            for epoch in range(self.epochs_trained+1, self.num_epochs+1):

                start_time = time.time()
                
                inner_corpus, inner_corpus_states, update_samples = \
                            self._subsample_corpus(
                                                corpus = corpus, 
                                                batch_svi = batch_svi, 
                                                locus_svi = locus_svi, 
                                                n_subsample_loci = n_subsample_loci
                                                )
                
                with TimerContext('E-step'):

                    sstats, new_gamma = self._inference(
                                parallel = parallel,
                                locus_subsample_rate=self.locus_subsample,
                                batch_subsample_rate=self.batch_size/self.n_samples,
                                learning_rate=learning_rate_fn(epoch),
                                corpus = inner_corpus,
                                model_state = self.model_state,
                                corpus_states = inner_corpus_states,
                                gamma = self._gamma[update_samples]
                            )

                self._gamma[update_samples, :] = new_gamma.copy()

                with TimerContext('M-step'):

                    self.model_state.update_state(sstats, learning_rate_fn(epoch))
                
                    if epoch >= 10 and self.empirical_bayes:
                        # wait 10 epochs to update the prior to prevent local minimas
                        for corpus_state in self.corpus_states.values():
                            corpus_state.update_alpha(sstats, learning_rate_fn(epoch))

                self.update_mutation_rates()

                elapsed_time = time.time() - start_time
                self.elapsed_times.append(elapsed_time)
                self.total_time += elapsed_time
                self.epochs_trained+=1

                if epoch % self.eval_every == 0:
                    
                    with TimerContext('Calculating bound'):
                        self.bounds.append(
                            self._bound(
                                corpus = corpus,
                                model_state = self.model_state,
                                corpus_states = self.corpus_states,
                                likelihood_scale=1,
                                gammas = self._gamma,
                            )
                        )

                    if len(self.bounds) > 1:
                        improvement = self.bounds[-1] - (self.bounds[-2] if epoch > 1 else self.bounds[-1])
                        
                        logger.info(f' [{epoch:>3}/{self.num_epochs+1}] | Time: {elapsed_time:<3.2f}s. '
                                    f'| Bound: { self.bounds[-1]:<10.2f}, improvement: {improvement:<10.2f} ')
                        
                elif not self.quiet:
                    logger.info(f' [{epoch:<3}/{self.num_epochs+1}] | Time: {elapsed_time:<3.2f}s. ')

                if not self.time_limit is None and self.total_time >= self.time_limit:
                    logger.info('Time limit reached, stopping training.')
                    break

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

        try:
            self._fit(corpus)
        except KeyboardInterrupt:
            logger.info('Training interrupted by user.')
            pass

        self.trained = True
        
        if not self.empirical_bayes:
            # if we're not using empirical bayes, we need to update the alpha parameters
            # to reflect the final state of the model.
            gammas = {name : [] for name in self.corpus_states.keys()}
            for sample, gamma in zip(corpus, self._gamma):
                gammas[sample.corpus_name].append(gamma)
            

            for corpus_state in self.corpus_states.values():
                corpus_state.set_alpha(np.array(gammas[corpus_state.corpus.name]))


        return self


    def partial_fit(self, corpus):
        return self._fit(corpus, reinit=False)


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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            return self._bound(
                    corpus = corpus, 
                    corpus_states = self.corpus_states,
                    model_state = self.model_state,
                    likelihood_scale=self.n_samples/n_samples,
                )


    @staticmethod
    def _ingest_vcf(self, vcf, corpus_state):
        
        return SBSSample.featurize_mutations(
            vcf_file = vcf,
            **corpus_state.corpus.instantiation_attrs
        )
    

    def _predict_sample(self, sample, corpus_name, n_gibbs_iters = 100):
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

        try:
            corpus_state = self.corpus_states[corpus_name]
        except KeyError:
            raise ValueError(f'Corpus {corpus_name} not found in model.')
        

        _, gamma = self._sample_inference(
            gamma0=self._init_doc_variables(1)[0],
            sample = sample,
            model_state = self.model_state,
            corpus_state = corpus_state,
        )

        '''observation_logits = IS._conditional_logp_mutation_locus(
            model_state= self.model_state,
            corpus_state = corpus_state,
            sample = sample,
        )

        z_posterior = IS._get_z_posterior(
            log_p_ml_z=observation_logits,
            alpha=corpus_state.alpha,
            n_iters=n_gibbs_iters,
            warm_up_steps=25,
        )'''

        return gamma #/gamma.sum()


    def predict(self, corpus):
        
        sample_names, gamma = [], []
        for sample in corpus:
            gamma.append(self._predict_sample(sample, corpus_name = sample.corpus_name))
            sample_names.append(sample.name)

        return sample_names, gamma



    def _calc_signature_sstats(self, corpus):

        _genome_trinuc_distribution = np.sum(
            corpus.trinuc_distributions,
            -1, keepdims= True
        )

        self._genome_trinuc_distribution = _genome_trinuc_distribution/_genome_trinuc_distribution.sum()



    def get_component_locus_distribution(self, corpus_name):
        '''
        For a sample, calculate the psi matrix - For each component, the distribution over loci.

        Parameters
        ----------

        Returns
        -------
        np.ndarray of shape (n_components, n_loci), where the entry for the k^th component at the l^th locus
            is the probability of sampling that locus given a mutation was generated by component k.

        '''
        return self.corpus_states[corpus_name]._log_mutation_rate


    def _pop_corpus(self, corpus):
        return next(iter(corpus))


    def get_expected_mutation_rate(self, corpus_name, sample = None):
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
        np.ndarray of shape (n_loci, ), The ln(probability) of sampling that locus given the processes
            in a sample.

        '''

        if not self.trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        psi_matrix = np.exp( self.get_component_locus_distribution(corpus_name) )

        if sample is None:
            corpus_state = self.corpus_states[corpus_name]
            gamma = corpus_state.alpha/corpus_state.alpha.sum() # use expectation of the prior over components
        else:
            gamma = self._predict(sample, corpus_name)

        return np.log( np.squeeze(np.dot(gamma, psi_matrix)) )
    

    def assign_sample_to_corpus(self, sample,
            n_samples_per_iter = 100, 
            n_iters = 100,
            max_mutations = 10000,
            pi_prior = None
        ):
        
        if not self.trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        logp_corpus = []; corpus_weights = []
        for corpus_name, corpus_state in self.corpus_states.items():
            
            logger.info(f'Testing hypothesis: mutations generated by {corpus_name}')

            observation_logits = IS._conditional_logp_mutation_locus(
                    model_state= self.model_state,
                    corpus_state = corpus_state,
                    sample = sample,
                )
            
            if observation_logits.shape[1] > max_mutations:
                logger.warn(f'Number of mutations in sample {sample.name} exceeds max_mutations, downsampling to {max_mutations} mutations.')
                sampled_mutations = np.random.RandomState(0).choice(observation_logits.shape[1], size = max_mutations, replace = False)
                observation_logits = observation_logits[:, sampled_mutations]

            if pi_prior is None:
                _alpha = corpus_state.alpha
            else:
                _alpha = np.ones_like(corpus_state.alpha) * pi_prior

            ais_weights = IS._annealed_importance_sampling(
                    log_p_ml_z=observation_logits,
                    alpha= _alpha,
                    n_iters=n_iters,
                    n_samples_per_iter=n_samples_per_iter
                )
            
            ais_weights = [w.sum() for w in ais_weights]

            logp_corpus.append(logsumexp(ais_weights) - np.log(len(ais_weights)))
            corpus_weights.append(ais_weights)

        posterior = np.exp(logp_corpus - logsumexp(logp_corpus))

        return {
            'corpus' : list(self.corpus_states.keys()),
            'log_marginal_likelihood' : list(logp_corpus),
            'posterior_probability' : list(posterior),
            'std_AIS' : [np.std(w) for w in corpus_weights]
        }



    def assign_mutations_to_components(self, sample, corpus_name, n_gibbs_iters = 1000,
                                       algorithm = 'gibbs'):

        if not self.trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        corpus_state = self.corpus_states[corpus_name]

        if algorithm == 'gibbs':
            observation_logits = IS._conditional_logp_mutation_locus(
                model_state= self.model_state,
                corpus_state = corpus_state,
                sample = sample,
            )

            z_posterior = IS._get_z_posterior(
                log_p_ml_z=observation_logits,
                alpha=corpus_state.alpha,
                n_iters=n_gibbs_iters,
            )
        elif algorithm == 'vi':
            pass
        else:
            raise ValueError(f'Algorithm {algorithm} not recognized.')


        MAP = np.argmax(z_posterior, axis = 0)

        return {
            'chrom' : sample.chrom,
            'pos' : sample.pos,
            'cosmic_str' : sample.cosmic_str,
            'region' : sample.locus,
            'MAP_assignment' : np.array(self.component_names)[MAP],
            **{
                'logP_' + component_name : np.log(z_posterior[k, :])
                for k, component_name in enumerate(self.component_names)
            }
        }

    
    def assign_mutations_to_corpus(self, sample, 
            n_samples_per_iter = 100, 
            n_iters = 100,
            max_mutations = 10000,
        ):
        
        if not self.trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        
        logp_corpus = []; corpus_names = np.array(list(self.corpus_states.keys()))
        for corpus_name, corpus_state in self.corpus_states.items():
            
            logger.info(f'Testing hypothesis: mutations generated by {corpus_name}')

            observation_logits = IS._conditional_logp_mutation_locus(
                model_state= self.model_state,
                corpus_state = corpus_state,
                sample = sample,
            )

            mutation_weights = IS._annealed_importance_sampling(
                    log_p_ml_z=observation_logits,
                    alpha=corpus_state.alpha,
                    n_iters=n_iters,
                    n_samples_per_iter= n_samples_per_iter
                )
            
            mutation_weights = np.array(mutation_weights) # N_iter x mutations
            
            #
            # mutation_weights are the logp of that mutation under the corpus
            # generative model for each annealing step. We take the average over
            # the annealing steps to get the logp of that mutation under the corpus.
            #

            logp_corpus.append(
                logsumexp(mutation_weights, axis = 0, keepdims = True) - np.log(mutation_weights.shape[0])
            )

        logp_corpus = np.concatenate(logp_corpus, axis = 0)
        
        MAP = np.argmax(logp_corpus, axis = 0)

        return {
            'chrom' : sample.chrom,
            'pos' : sample.pos,
            'cosmic_str' : sample.cosmic_str,
            'region' : sample.locus,
            'MAP_assignment' : corpus_names[MAP],
            **{
                'logP_' + _corpus_name : logp_corpus[k, :]
                for k, _corpus_name in enumerate(corpus_names)
            }
        }
    


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
            return list(mean)


    def plot_signature(self, component, ax = None, 
                       figsize = (5.5,1.25), 
                       normalization = 'global', 
                       fontsize=7):
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
            linewidth = 0.5,
            yerr = error,
            error_kw = {'alpha' : 0.5, 'linewidth' : 0.5}
        )

        for s in ['left','right','top']:
            ax.spines[s].set_visible(False)

        ax.set(yticks = [], xticks = [], 
               xlim = (0,96), ylim = (0, 1.1*max(mean + error)))
        ax.set_title('Component ' + str(component), fontsize = fontsize)

        return ax


    def _order_features(self):
        
        feature_linkage = hierarchy.linkage(self.model_state.beta_mu.T)

        feature_num_order = hierarchy.leaves_list(
                            hierarchy.optimal_leaf_ordering(
                                feature_linkage, 
                                self.model_state.beta_mu.T
                            )
                        )
        
        feature_order = self.feature_names[feature_num_order]

        return feature_order


    def _cluster_component_associations(self, cluster_distance):

        component_linkage = hierarchy.linkage(self.model_state.beta_mu)

        clusters = hierarchy.fcluster(component_linkage, cluster_distance, 
                                      criterion='distance') - 1
        
        cluster_ids, counts = np.unique(clusters, return_counts=True)

        cluster_order = cluster_ids[np.argsort(counts)[::-1]]

        cluster_order_map = dict(zip(cluster_order, range(len(cluster_ids))))
        clusters = [cluster_order_map[c] for c in clusters]

        return clusters


    def plot_coefficients(self, component, 
        ax = None, figsize = None, 
        error_std_width = 5,
        dotsize = 10,
        color = 'black',
        reorder_features = True,
        fontsize = 8,
    ):
        '''
        Plot regression coefficients with 99% confidence interval.
        '''

        assert isinstance(component, int) and 0 <= component < self.n_components

        if ax is None:
            if figsize is None:
                figsize = (self.n_locus_features/3, 1.25)

            _, ax = plt.subplots(1,1,figsize= figsize)

        mu, nu = self.model_state.beta_mu[component,:], self.model_state.beta_nu[component,:]

        std_width = error_std_width/2
        ax.bar(
            x=self.feature_names,
            height=mu,
            yerr=std_width*nu,
            color=['darkred' if _mu < 0 else color for _mu in mu],
            edgecolor='black',
            linewidth=0.1,  # Adjust the linewidth here
            width = 0.33,
            capsize=3,
            zorder=4,
            alpha = 0.75,
        )

        ax.axhline(0, linestyle='--', color='lightgrey', linewidth=0.5,
                    zorder = 0)

        ax.xaxis.grid(True, color = 'lightgrey', linestyle = '--', linewidth = 0.5,
                        zorder = 0)
        
        #// remove the ticks from the y axis but not the labels
        ax.tick_params(axis='x', which='both', length=0, labelsize = fontsize,
                        rotation = 90)
        ax.tick_params(axis='y', which='both', labelsize = fontsize)

        ax.set(xlabel = 'Feature', ylabel = 'Coefficient')
        ax.xaxis.label.set_size(fontsize); ax.yaxis.label.set_size(fontsize)

        ax.set_title(self.component_names[component], fontsize = fontsize)

        for s in ['left','right','top', 'bottom']:
            ax.spines[s].set_visible(False)

        if reorder_features:
            ax.set_xticks(self._order_features())

        return ax



    def plot_compare_coefficients(self, ax = None, figsize = None,
                              reorder_features = True,
                              cluster_components = True,
                              cluster_distance = 0.75,
                              mask_largest_cluster = True,
                              palette = 'tab10',
                              clusters = None,
                              dotsize = 15,
                              fontsize = 7,
                            ):
        
        if ax is None:
            if figsize is None:
                figsize = (2*self.n_locus_features/3, 2*1)

            _, ax = plt.subplots(1,1,figsize= figsize)

        if cluster_components:
            
            if isinstance(palette, str):
                color_list = list(plt.get_cmap(palette).colors)
            else:
                color_list = palette

            if clusters is None:
                clusters = self._cluster_component_associations(cluster_distance)
                
                if max(set(clusters)) > 1 and mask_largest_cluster:
                    color_list = ['grey'] + color_list

                assert len(clusters) <= len(color_list), \
                    f'Number of clusters ({len(clusters)}) exceeds number of colors ({len(color_list)}).\n' \
                    'Try using a larger palette, or increasing the cluster_distance paramter.'
                

            component_colors = [color_list[cluster] for cluster in clusters]
            num_clusters = len(set(clusters))

            cluster_membership = defaultdict(list)
            for component, cluster in enumerate(clusters):
                cluster_membership[cluster].append(component)
            
            ax.legend(
                [plt.Circle((0,0),0.5, color = color) for color in color_list[:num_clusters]],
                ['Component: ' + ', '.join([str(c) for c in cluster_membership[cluster]]) for cluster in set(clusters)],
                fontsize = fontsize,
                loc = 'upper left',
                bbox_to_anchor=(1.01, 1.0),
                frameon = False,
                title = 'Cluster',
                title_fontsize = fontsize,
                markerscale = 0.6
            )
            

        else:
            component_colors = ['black']*self.n_components


        for component, color in zip(range(self.n_components), component_colors):

            self.plot_coefficients(component, 
                                   ax = ax, 
                                   color = color,
                                   dotsize=dotsize,
                                   reorder_features = False,
                                   fontsize=fontsize,
                                   )

        if reorder_features:
            ax.set_xticks(self._order_features())
        
        ax.set(title = '')
        
        return ax


    def plot_summary(self, fontsize = 7):
        
        coefplot_width = self.n_locus_features/3
        figwidth = coefplot_width + 5.5

        _, ax = plt.subplots(
                            self.n_components, 2, 
                            figsize = (figwidth, 1.25*self.n_components),
                            sharex = 'col',
                            gridspec_kw={
                                'width_ratios': [coefplot_width, figwidth - coefplot_width]
                                }
                            )
        
        for i in range(self.n_components):

            self.plot_signature(i, ax = ax[i,1], normalization='global', fontsize=fontsize)
            self.plot_coefficients(i, ax = ax[i,0], fontsize=fontsize)
            ax[i,0].set(xlabel = '', ylabel = self.component_names[i], title = '')
            ax[i,1].set(title = '')
            ax[i,0].spines['bottom'].set_visible(False)

        ax[-1, 0].set(xlabel = 'Feature')

        return ax


