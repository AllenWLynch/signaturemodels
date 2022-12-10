import numpy as np
from .base import *
import logging
import tqdm
import warnings
logging.basicConfig(level=logging.INFO,
                    format='')
logger = logging.getLogger(__name__)
import time

def initialize_parameters(
        pi_prior = 1., 
        m_prior = 1., 
        beta_prior = 1.,*,
        dtype, 
        n_components, 
        n_contexts):
    
    alpha = (np.ones((n_components)) * pi_prior).astype(dtype, copy = False)
    
    b = (np.ones((n_components,2,3)) * m_prior).astype(dtype, copy = False)
    
    nu = (np.ones((2, n_contexts//2)) * beta_prior).astype(dtype, copy = False)
    
    return b, alpha, nu


def initialize_variational_parameters(*,random_state, dtype, n_components, n_contexts):
    
    
    _lambda = random_state.gamma(100, 1./100., 
                    (n_components, 2, n_contexts//2) # n_components, two mut categories, n_contexts
        ).astype(dtype, copy = False)
    
    
    epsilon = random_state.gamma(100., 1./100., 
                              (n_components, 2, n_contexts//2, 3) # n_components, two mut categories, n_contexts, n_mutations
                            ).astype(dtype, copy=False)
    
    return epsilon, _lambda


def initialize_document_variational_parameters(*,
        random_state, dtype, n_samples, n_components):

    gamma = random_state.gamma(100., 1./100., 
                               (n_samples, n_components) # n_genomes by n_components
                              ).astype(dtype, copy=False)
    
    return gamma


def M_step_alpha(*, alpha, gamma, rho, optimize):
    
    N = gamma.shape[0]
    log_phat = log_dirichlet_expectation(gamma).mean(-2)
    
    return update_dir_prior(alpha, N, log_phat, rho = rho, optimize = optimize)


def E_step_phi(gamma, phi_matrix_prebuild, freq_matrix):

    logE_gamma = log_dirichlet_expectation(gamma)

    phi_matrix_unnormalized = phi_matrix_prebuild * np.exp(logE_gamma)[:,:,None,None,None]
    phi_matrix = phi_matrix_unnormalized/phi_matrix_unnormalized.sum(1, keepdims = True)

    weighted_phi = phi_matrix*np.expand_dims(freq_matrix, 1)

    return phi_matrix, weighted_phi


class LdaModel(BaseModel):

    n_contexts = 32

    def _bound(self, gamma, phi_matrix, weighted_phi, likelihood_scale = 1.):

        logE_gamma = log_dirichlet_expectation(gamma)
        evidence = 0

        # 
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='divide by zero encountered in log')

            evidence += np.sum(weighted_phi*(
                    np.expand_dims(self.logE_epsilon, 0) + logE_gamma[:,:,None,None,None] \
                        + self.logE_lambda[None,:,:,:,None] - np.where(phi_matrix > 0, np.log(phi_matrix), 0.)
                    )
                )
        
        evidence += sum(dirichlet_bound(self.alpha, gamma, logE_gamma))

        evidence*=likelihood_scale
        
        for i in range(self.n_components):
            for j in [0,1]:
                evidence+=sum(
                    dirichlet_bound(self.b[i,j], self.epsilon[i,j], self.logE_epsilon[i,j])
                )
        
        evidence+=sum(
            dirichlet_bound(
                self.nu.reshape(-1), 
                self._lambda.reshape(self.n_components,-1), 
                self.logE_lambda.reshape(self.n_components,-1)
            )
        )

        return evidence


    def _inference(self, gamma, freq_matrix, 
        difference_tol = 0.001, iterations = 100, quiet = True):

        phi_matrix_prebuild = np.exp(self.logE_lambda)[:,:,:,None] * np.exp(self.logE_epsilon)

        # Do the first E-step out of the loop
        old_gamma = gamma.copy()
        phi_matrix, weighted_phi = E_step_phi(
                    old_gamma, 
                    phi_matrix_prebuild,
                    freq_matrix
                )
        gamma = self.alpha + weighted_phi.sum(axis = (-1,-2,-3))

        not_converged = np.abs(gamma - old_gamma).mean(axis = 1) > difference_tol

        for i in range(iterations) if quiet \
            else tqdm.tqdm(range(iterations), desc = 'E-step'):
            
            if np.all(~not_converged):
                logging.debug('\tStopped E-step after {} iterations.'.format(str(i+1)))
                break
            else:

                old_gamma = gamma[not_converged].copy()

                phi_matrix[not_converged], weighted_phi[not_converged] = E_step_phi(
                            old_gamma, 
                            phi_matrix_prebuild,
                            freq_matrix[not_converged]
                        )

                gamma[not_converged] = self.alpha + weighted_phi[not_converged]\
                                                    .sum(axis = (-1,-2,-3))

                not_converged[not_converged] = np.abs(gamma[not_converged] - old_gamma).mean(1) > difference_tol

        else:
            logging.info('\t{} samples reached maximum E-step iterations.'\
                    .format(str(not_converged.sum())))

        return gamma, phi_matrix, weighted_phi


    @extract_freqmatrix
    def fit(self, freq_matrix):

        assert isinstance(self.n_components, int) and self.n_components > 1

        self.random_state = np.random.RandomState(self.seed)

        self.b, self.alpha, self.nu = initialize_parameters(
            pi_prior= self.pi_prior, 
            m_prior= self.m_prior, 
            beta_prior= self.beta_prior,
            dtype = self.dtype,
            n_components = self.n_components,
            n_contexts=self.n_contexts
        )

        self.epsilon, self._lambda = initialize_variational_parameters(
            random_state=self.random_state, dtype=self.dtype,
            n_components=self.n_components, n_contexts=self.n_contexts,
        )

        gamma = initialize_document_variational_parameters(
            random_state = self.random_state, dtype = self.dtype, 
            n_components = self.n_components,
            n_samples= len(freq_matrix),
        )

        corpus_size = len(freq_matrix)
        batch_size = min(self.batch_size, corpus_size)
        sstat_scale = corpus_size/batch_size
        batch_lda = batch_size == corpus_size

        logger.debug('Sstat scale: {}'.format(str(sstat_scale)))
        logger.debug('Batch size: {}'.format(str(batch_size)))

        if batch_lda:
            logger.warn('Running batch LDA algorithm')

        self.bounds = []

        _it = range(self.num_epochs)
        #if not self.quiet:
        #    _it = tqdm.tqdm(_it, desc = 'Training')
        improvement_ema = None
        start_time = time.time()

        try:
            for epoch in _it:
                
                epoch_start_time = time.time()
                logger.info('Begin Epoch {} E-step.'.format(str(epoch)))

                rho = 1. if batch_lda else self.get_rho(epoch)

                if not batch_lda:
                    logger.debug('\tRho: {:.3f}'.format(rho))

                if not batch_lda:
                    subsample = np.random.choice(corpus_size, 
                            size = batch_size, 
                            replace = False
                        )
                else:
                    subsample = np.arange(corpus_size)
                
                #logger.debug('\tSubsample size: {}'.format(str(len(subsample))))

                gamma[subsample], phi_matrix, weighted_phi = \
                    self._inference(
                        gamma[subsample], 
                        freq_matrix[subsample], 
                        difference_tol = self.difference_tol,
                        iterations = self.estep_iterations
                    )

                if ~np.all(np.isfinite(weighted_phi)):
                    raise ValueError('Non-finite values detected in sufficient statistics. Stopping training to preserve current state.')

                logger.debug('\tBegin M-step.'.format(str(epoch)))

                self.epsilon, self._lambda = \
                    self._estimate_global_parameters(
                            weighted_phi, 
                            rho, 
                            sstat_scale=sstat_scale, 
                    )
                
                if epoch > 0 and epoch % self.prior_update_every == 0:

                    logger.debug('\tUpdating priors.')
                    # local prior update
                    self.alpha = self.svi_update(
                        self.alpha, 
                        M_step_alpha(
                            gamma = gamma[subsample],
                            alpha = self.alpha,  
                            rho = rho,
                            optimize = batch_lda,
                        ), 
                        rho
                    )

                    self.b, self.nu = \
                        self._estimate_global_priors(
                                rho,
                                optimize = batch_lda,
                        )
                        

                if epoch % self.eval_every == 0:
                    
                    if not batch_lda:
                        phi_matrix, weighted_phi = E_step_phi(
                            gamma, 
                            np.exp(self.logE_lambda)[:,:,:,None] * np.exp(self.logE_epsilon) , 
                            freq_matrix
                        )

                    self.bounds.append(
                        self._bound(gamma = gamma, phi_matrix = phi_matrix,
                            weighted_phi=weighted_phi)
                    )

                    if not np.isfinite(self.bounds[-1]):
                        logger.warn('\tBound is not finite on training data, stopping training.')
                        break

                    elif len(self.bounds) > 1:

                        improvement = self.bounds[-1] - self.bounds[-2]
                        logger.info('\tBounds improvement: {:.2f}'.format(improvement))

                        improvement_ema = (1-0.1) * improvement_ema + 0.1*improvement
                        logger.debug('\tImprovement EMA: {:.2f}'.format(-improvement_ema))
                        if 0 < -improvement_ema < self.bound_tol:
                            break
                        
                    else:
                        improvement_ema = self.bounds[-1]

                logger.info('\tEpoch time: {:.2f} sec, total elapsed time {:.1f} min'\
                    .format(
                        time.time() - epoch_start_time,
                        (time.time() - start_time)/60
                    ))

        except KeyboardInterrupt:
            pass
        
        return self


    def _infer_document_variables(self, freq_matrix):

        return self._inference(
                initialize_document_variational_parameters(
                    random_state=self.random_state,
                    dtype=self.dtype,
                    n_samples=len(freq_matrix),
                    n_components=self.n_components
                ), 
                freq_matrix, 
                difference_tol = 1e-3, 
                iterations = 10000,
                quiet=False,
            )

    @extract_freqmatrix
    def predict(self, freq_matrix):

        return  np.exp(log_dirichlet_expectation(
            self._infer_document_variables(freq_matrix)[0]
        ))


    @extract_freqmatrix
    def score(self, freq_matrix):

        return self._bound(
            *self._infer_document_variables(freq_matrix)
        )/np.sum(freq_matrix) # per-word perplexity