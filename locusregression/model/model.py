
from ._dirichlet_update import log_dirichlet_expectation, pseudo_r2, dirichlet_bound
from locusregression.corpus import COSMIC_SORT_ORDER, SIGNATURE_STRINGS, MUTATION_PALETTE
from locusregression.corpus.sbs_sample import SBSSample
import locusregression.model._sstats as _sstats
from ._model_state import ModelState, CorpusState
from ._importance_sampling import _get_z_posterior
from pandas import DataFrame
# external imports
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import pickle
import logging
logger = logging.getLogger(' LocusRegressor')



def _get_observation_likelihood(*,model_state, sample, corpus_state):
    '''
    Flattened phi is the probability of a mutation and locus for each signature given the current modelstate.
    Shape: N_sigs x N_mutations
    '''
    rho = model_state.omega/model_state.omega.sum(axis = -1, keepdims = True)
    
    flattend_phi = (
            np.log(rho) + \
            np.log(model_state.delta)[:,:,None]
        )[:, sample.context, sample.mutation]\
        + np.log(sample.exposures[:, sample.locus]) \
        + np.log(corpus_state.trinuc_distributions[sample.context, sample.locus]) \
        + corpus_state.logmu[:, sample.locus] \
        - corpus_state.get_log_denom(sample.exposures)\
    
    exp_phi = np.nan_to_num(np.exp(flattend_phi), nan=0)

    if not np.all(np.isfinite(exp_phi)):
        print('problem!')

    return exp_phi


def _estep_update(exp_Elog_gamma, alpha, flattend_phi, count_g, likelihood_scale = 1):
    gamma_sstats = exp_Elog_gamma*np.dot(flattend_phi, count_g/np.dot(flattend_phi.T, exp_Elog_gamma))
    gamma_sstats = gamma_sstats.reshape(-1)
    return alpha + gamma_sstats*likelihood_scale


def _calc_local_variables(*,gamma, flattend_phi):
    exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma)[:,None])
    phi_matrix = np.outer(exp_Elog_gamma, 1/np.dot(flattend_phi.T, exp_Elog_gamma))*flattend_phi #/(batch_subsample_rate*locus_subsample_rate)
    
    return phi_matrix


class TimerContext:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        logger.info(f'{self.name} - time: {elapsed_time:<3.2f}s. ')


class LocusRegressor:

    MODEL_STATE = ModelState
    CORPUS_STATE = CorpusState
    SSTATS = _sstats
    
    n_contexts = 32

    @classmethod
    def sample_params(cls, trial):
        return dict()
        

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    

    def __init__(self, 
        n_components = 10,
        seed = 2, 
        dtype = float,
        pi_prior = 1.,
        num_epochs = 300, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        quiet = False,
        n_jobs = 1,
        locus_subsample = 1,
        batch_size = None,
        empirical_bayes = True,
        kappa = 0.5,
        tau = 16,
        eval_every = 50,
        begin_prior_updates = 10,
        time_limit = None,
        fix_signatures =  None,
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
        self.begin_prior_updates = begin_prior_updates


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    
    @staticmethod
    def _perplexity(elbo, corpus):
        return np.exp(-elbo/corpus.num_mutations)


    @staticmethod
    def _sample_bound(
            sample,
            model_state,
            corpus_state,
            gamma
        ):

        flattened_phi = _get_observation_likelihood(
                model_state=model_state,
                sample=sample,
                corpus_state=corpus_state,
            )
        
        phi = _calc_local_variables(
            gamma = gamma, 
            flattend_phi=flattened_phi
        )

        weighted_phi = phi*np.array(sample.weight)[None,:]

        entropy_sstats = -np.sum(weighted_phi * np.where(phi > 0, np.log(phi), 0.))
        entropy_sstats += np.sum(dirichlet_bound(corpus_state.alpha, gamma))
        
        flattened_logweight = log_dirichlet_expectation(gamma)[:,None] + np.nan_to_num(np.log(flattened_phi), nan=-np.inf)
        
        return np.sum(weighted_phi*flattened_logweight) + entropy_sstats


    def _bound(self,*,
            corpus, 
            model_state,
            corpus_states,
            gamma = None,
        ):

        elbo = 0
        for sample, _gamma in zip(corpus, gamma):
            elbo += self._sample_bound(
                sample=sample,
                model_state=model_state,
                corpus_state=corpus_states[sample.corpus_name],
                gamma=_gamma,
            )

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
        
        flattend_phi = _get_observation_likelihood(
                model_state=model_state,
                sample=sample,
                corpus_state=corpus_state
            )

        count_g = np.array(sample.weight)[:,None].astype(self.dtype, copy=False)
        gamma = gamma0.copy()
        
        for _ in range(self.estep_iterations): # inner e-step loop
            
            old_gamma = gamma.copy()
            exp_Elog_gamma = np.exp(log_dirichlet_expectation(gamma)[:,None])

            gamma = _estep_update(exp_Elog_gamma, corpus_state.alpha, flattend_phi, count_g,
                                 likelihood_scale=1/locus_subsample_rate)

            if np.abs(gamma - old_gamma).mean() < self.difference_tol:
                break
        else:
            logger.debug('E-step did not converge. If this happens frequently, consider increasing "estep_iterations".')

        gamma = (1 - learning_rate)*gamma0 + learning_rate*gamma

        phi_matrix = _calc_local_variables(gamma = gamma, flattend_phi=flattend_phi)
        weighted_phi = phi_matrix * count_g.T/(locus_subsample_rate * batch_subsample_rate)

        return self.SSTATS.SampleSstats(
            model_state=model_state,
            sample=sample,
            weighted_phi=weighted_phi,
            gamma=gamma,
        )

    
    def _inference(self,
            locus_subsample_rate=1,
            batch_subsample_rate=1,
            learning_rate=1,*,
            corpus,
            model_state,
            corpus_states,
            gamma,
        ):
        
        sstat_collections = {
            corp.name : self.SSTATS.CorpusSstats(model_state) 
            for corp in corpus.corpuses
        }

        gammas = []
        for gamma_g, sample in zip(gamma, corpus):

            sample_sstats = self._sample_inference(
                        sample = sample,
                        gamma0 = gamma_g, 
                        locus_subsample_rate = locus_subsample_rate,
                        batch_subsample_rate = batch_subsample_rate,
                        learning_rate = learning_rate,
                        model_state = model_state,
                        corpus_state = corpus_states[sample.corpus_name]
                    )

            sstat_collections[sample.corpus_name] += sample_sstats
            gammas.append(sample_sstats.gamma)

        return self.SSTATS.MetaSstats(sstat_collections), np.vstack(gammas)
        

    def _update_mutation_rates(self):
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
    
    @property
    def is_trained(self):
        try:
            return self._trained
        except AttributeError:
            return False

    def _init_model(self, corpus):

        self._trained = False

        self.random_state = np.random.RandomState(self.seed)
        self.bound_random_state = np.random.RandomState(self.seed + 1)

        self.n_locus_features, self.n_loci = corpus.shape
        self.n_samples, self.feature_names = len(corpus), corpus.feature_names

        self._calc_signature_sstats(corpus)

        if not self.fix_signatures is None:
            logger.warn('Initializing signatures: ' + ', '.join(map(str, self.fix_signatures)) + ', will still be updated during training.')

        
        fix_strs = list(self.fix_signatures) if not self.fix_signatures is None else []
        self.component_names = fix_strs + ['Component ' + str(i) for i in range(self.n_components - len(fix_strs))]

        self.training_bounds_ = []; self.testing_bounds_ = []
        self.elapsed_times = []
        self.total_time = 0
        self.epochs_trained = 0
        
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

        self.model_state = self.MODEL_STATE(
                n_components=self.n_components,
                random_state=self.random_state,
                n_features=self.n_locus_features,
                n_loci=self.n_loci,
                empirical_bayes = self.empirical_bayes,
                dtype = self.dtype,
                fix_signatures = self.fix_signatures,
                genome_trinuc_distribution = self._genome_trinuc_distribution,
                corpus_states=self.corpus_states,
                **self._get_rate_model_parameters()
        )

        self._gamma = self._init_doc_variables(self.n_samples)

        for state in self.corpus_states.values():
            state.update_log_denom(self.model_state, state.exposures)


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



    def _fit(self,corpus, 
                  reinit = True, 
                  test_corpus = None, 
                  subset_by_loci = False
            ):
        
        self.batch_size = len(corpus) if self.batch_size is None else min(self.batch_size, len(corpus))
        self.locus_subsample = self.locus_subsample or 1
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

        
        with warnings.catch_warnings():
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

                    self.model_state.update_state(sstats, inner_corpus_states, learning_rate_fn(epoch))
                
                    if epoch >= self.begin_prior_updates and self.empirical_bayes:
                        # wait some # of epochs to update the prior to prevent local minimas
                        for corpus_state in self.corpus_states.values():
                            corpus_state.update_alpha(sstats, learning_rate_fn(epoch))

                self._update_mutation_rates()

                elapsed_time = time.time() - start_time
                self.elapsed_times.append(elapsed_time)
                self.total_time += elapsed_time
                self.epochs_trained+=1

                if (not self.eval_every is None) and epoch % self.eval_every == 0:
                    
                    with TimerContext('Calculating bound'):
                        
                        elbo = self._bound(
                                corpus = corpus,
                                model_state = self.model_state,
                                corpus_states = self.corpus_states,
                                gamma = self._gamma,
                            )
                    self.training_bounds_.append(self._perplexity(elbo, corpus))

                    if not test_corpus is None:
                        self.testing_bounds_.append(
                            self.score(test_corpus, subset_by_loci = subset_by_loci)
                        )

                    if len(self.training_bounds_) > 1:
                        improvement = self.training_bounds_[-1] - (self.training_bounds_[-2] if epoch > 1 else self.training_bounds_[-1])
                    else:
                        improvement = 0

                    logger.info(f' [{epoch:>3}/{self.num_epochs+1}] | Time: {elapsed_time:<3.2f}s. '
                                f'| Perplexity: { self.training_bounds_[-1]:<10.2f}, improvement: {-improvement:<10.2f} ')

                    if test_corpus is None:
                        yield (self.training_bounds_[-1], np.nan)
                    else:
                        yield (self.training_bounds_[-1], self.testing_bounds_[-1])
                        
                elif not self.quiet:
                    logger.info(f' [{epoch:<3}/{self.num_epochs+1}] | Time: {elapsed_time:<3.2f}s. ')

                if not self.time_limit is None and self.total_time >= self.time_limit:
                    logger.info('Time limit reached, stopping training.')
                    break



    def fit(self, corpus, test_corpus = None, subset_by_loci = False):
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
            for _ in self._fit(
                corpus, 
                reinit=True, 
                test_corpus = test_corpus, 
                subset_by_loci = subset_by_loci
            ):
                pass
        except KeyboardInterrupt:
            logger.info('Training interrupted by user.')
            pass

        self._trained = True

        return self


    def partial_fit(self, corpus):
        return self._fit(corpus, reinit=False)


    def _init_new_corpusstates(self, corpus):

        corpus_state_clones = {
            name : state.clone_corpusstate(corpus.get_corpus(name))
            for name, state in self.corpus_states.items()
            if name in corpus.corpus_names
        }
        
        for clone in corpus_state_clones.values():
            clone.update_mutation_rate(self.model_state, from_scratch = True)

        return corpus_state_clones


    def score(self, corpus, subset_by_loci = False):
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

        if not self.is_trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        corpus_state_clones = self._init_new_corpusstates(corpus)
        
        if not subset_by_loci:
            gamma = np.array([
                self._predict_sample(sample, corpus_state_clones[sample.corpus_name]) for sample in corpus
            ])
        else:
            logger.warn('`subset_by_loci` was TRUE, so using the predicted gamma values for the corpus.')
            gamma = self._gamma


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            elbo = self._bound(
                    corpus = corpus, 
                    corpus_states = corpus_state_clones,
                    model_state = self.model_state,
                    gamma=gamma,
                )
            
        return self._perplexity(elbo, corpus)



    @staticmethod
    def _ingest_vcf(self, vcf, corpus_state):
        
        return SBSSample.featurize_mutations(
            vcf_file = vcf,
            **corpus_state.corpus.instantiation_attrs
        )
    

    def _predict_sample(self, sample, corpus_state):
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

        gamma = self._sample_inference(
            gamma0=self._init_doc_variables(1)[0],
            sample = sample,
            model_state = self.model_state,
            corpus_state = corpus_state,
        ).gamma

        return gamma


    def predict(self, corpus):
        """
        Predicts the gamma values for the given corpus.

        Parameters
        ----------
        corpus (list): List of samples to predict gamma values for.

        Returns
        -------
        tuple: A tuple containing the sample names and their corresponding gamma values.
        """        
        if not self.is_trained:
            logger.warn('This model was not trained to completion, results may be inaccurate')

        corpus_state_clones = self._init_new_corpusstates(corpus)
        
        sample_names, gamma = [], []
        for sample in corpus:
            gamma.append(self._predict_sample(
                            sample, 
                            corpus_state_clones[sample.corpus_name]
                            )
                        )
            sample_names.append(sample.name)

        return DataFrame(
            np.vstack(gamma),
            index = sample_names,
            columns = self.component_names
        )


    def _calc_signature_sstats(self, corpus):

        _genome_trinuc_distribution = np.sum(
            corpus.trinuc_distributions,
            -1, keepdims= True
        )

        self._genome_trinuc_distribution = _genome_trinuc_distribution/_genome_trinuc_distribution.sum()


    def get_log_nucleotide_effect(self, corpus):

        try:
            self.corpus_states[corpus.name]
        except KeyError:
            raise ValueError(f'Corpus {corpus.name} not found in model.')
        
        return np.log( self.model_state.delta @ corpus.trinuc_distributions )


    def get_log_component_mutation_rate(self, corpus):
        '''
        For a sample, calculate the psi matrix - For each component, the distribution over loci.

        Parameters
        ----------

        Returns
        -------
        np.ndarray of shape (n_components, n_loci), where the entry for the k^th component at the l^th locus
            is the probability of sampling that locus given a mutation was generated by component k.

        '''
        try:
            corpus_state = self.corpus_states[corpus.name]
        except KeyError:
            raise ValueError(f'Corpus {corpus.name} not found in model.')
        
        new_state = corpus_state.clone_corpusstate(corpus)
        new_state.update_mutation_rate(self.model_state, from_scratch = True)

        return new_state.get_log_component_effect_rate(
                    self.model_state, new_state.exposures
                )


    def _pop_corpus(self, corpus):
        return next(iter(corpus))


    def get_log_marginal_mutation_rate(self, corpus):
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
        if not self.is_trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        corpus_state = self.corpus_states[corpus.name]
        gamma = corpus_state.alpha/corpus_state.alpha.sum() # use expectation of the prior over components
        
        psi_tensor = np.exp(self.get_log_component_mutation_rate(corpus)) # n_components x n_contexts x n_loci
        
        marginalized = np.squeeze(np.tensordot(gamma, psi_tensor, axes=([0], [0])))

        return np.log(marginalized)
    
    
    def get_mutation_rate_r2(self, corpus):
        '''
        Calculate the deviance of the mutation rate for a given corpus.

        Parameters:
        corpus (Corpus): The corpus for which to calculate the deviance.

        Returns:
        float: The deviance of the mutation rate.
        '''

        if not self.is_trained:
            logger.warn('This model was not trained to completion, results may be innaccurate')

        try:
            self.corpus_states[corpus.name]
        except KeyError:
            raise ValueError(f'Corpus {corpus.name} not found in model.')
        
        empirical_mr = corpus.get_empirical_mutation_rate().toarray()
        predicted_mr = np.exp(self.get_log_marginal_mutation_rate(corpus))
        y_null = corpus.trinuc_distributions

        return pseudo_r2(empirical_mr, predicted_mr, y_null)
    


    def _get_signature_idx(self, component):

        try:
            component = int(component)
        except ValueError:
            pass

        if isinstance(component, int):
            assert 0 <= component < self.n_components
        elif isinstance(component, str):
            try:
                component = self.component_names.index(component)
            except ValueError:
                raise ValueError(f'Component {component} not found in model.')
        else:
            raise ValueError(f'Component {component} not found in model.')
        
        return component
    

    def signature(self, component, 
            normalization = 'global'):
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
        component = self._get_signature_idx(component)
        
        try:
            if normalization == 'local':
                norm_frequencies = self._weighted_trinuc_dists[component]
            elif normalization == 'global':
                norm_frequencies = self._genome_trinuc_distribution.ravel()
            else:
                raise ValueError(f'Normalization option {normalization} does not exist.')
        except AttributeError as err:
            raise AttributeError('User must run "model.calc_signature_sstats(corpus)" before extracting signatures from data.') from err

        eps_expectation = self.model_state.omega[component]/self.model_state.omega[component].sum(axis = 1, keepdims = True) 
        lambda_expectation = self.model_state.delta[component]*norm_frequencies\
                                /(norm_frequencies*self.model_state.delta[component]).sum()

        signature_expectation = (lambda_expectation[:,None]*eps_expectation).reshape(-1)

        sig_dict = dict(zip(SIGNATURE_STRINGS, signature_expectation))

        return [sig_dict[mut] for mut in COSMIC_SORT_ORDER]



    def plot_signature(self, component, ax = None, 
                       figsize = (5.5,1.25), 
                       normalization = 'global', 
                       fontsize=7):
        '''
        Plot signature.
        '''

        component = self._get_signature_idx(component)

        mean = self.signature(
            component,
            normalization = normalization
        )
        
        if ax is None:
            _, ax = plt.subplots(1,1,figsize= figsize)

        ax.bar(
            height = mean,
            x = COSMIC_SORT_ORDER,
            color = MUTATION_PALETTE,
            width = 1,
            edgecolor = 'white',
            linewidth = 0.5,
            error_kw = {'alpha' : 0.5, 'linewidth' : 0.5}
        )

        for s in ['left','right','top']:
            ax.spines[s].set_visible(False)

        ax.set(yticks = [], xticks = [], 
               xlim = (-1,96), ylim = (0, 1.1* max(mean))
            )  
        ax.set_title(self.component_names[component], fontsize = fontsize)

        return ax


    def plot_summary(self):
        """
        Plot the summary of the model.

        Returns:
            ax (matplotlib.axes.Axes): The axes object containing the plot.
        """

        fig, ax = plt.subplots(self.n_components, 1, 
                               figsize=(5.5, 1.25*self.n_components), 
                               sharex=True
                               )

        for i in range(self.n_components):
            self.plot_signature(i, ax=ax[i])
            ax[i].set_title('')
            ax[i].set_ylabel(self.component_names[i], fontsize=7)

        return ax
    

    def get_empirical_component_mutation_rate(self, corpus):
        """
        Calculate the empirical component mutation rate for a given corpus.

        Parameters:
        corpus (Corpus): The corpus for which to calculate the empirical component mutation rate.

        Returns:
        numpy.ndarray: An array containing the empirical component mutation rate.
        """
        corpus_state_clones = self._init_new_corpusstates(corpus)
        
        sstats, _ = self._inference(
            locus_subsample_rate=1,
            batch_subsample_rate=1,
            learning_rate=1,
            corpus = corpus,
            model_state = self.model_state,
            corpus_states = corpus_state_clones,
            gamma = self._init_doc_variables(len(corpus))
        )

        return np.array([
            self.model_state._convert_beta_sstats_to_array(k, sstats, corpus.shape[1]).ravel()
            for k in range(self.n_components)
        ])
    


    def get_posterior_assignments(self, sample, corpus, n_iters = 1000):
        
        try:
            corpus_state = self.corpus_states[corpus.name]
        except KeyError:
            raise ValueError(f'Corpus {corpus.name} not found in model.')
        
        new_corpus_state = corpus_state.clone_corpusstate(corpus)
        new_corpus_state.update_mutation_rate(self.model_state, from_scratch = True)

        observation_ll = np.log(
            _get_observation_likelihood(
                model_state=self.model_state,
                sample=sample,
                corpus_state=new_corpus_state
            )
        )

        z_posterior = np.log(
            _get_z_posterior(
                observation_ll,
                alpha = new_corpus_state.alpha,
                randomstate = self.random_state,
                n_iters = n_iters
            )
        )

        df_cols = {
            'CHROM' : sample.chrom.astype(str),
            'POS' : sample.pos
        }

        df_cols.update({
            f"INFO/logp-{component_name.replace(' ','_')}" : _posterior
            for component_name, _posterior in zip(self.component_names, z_posterior)
        })

        return DataFrame(df_cols)
        


        