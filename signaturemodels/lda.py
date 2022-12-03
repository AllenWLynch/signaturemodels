import numpy as np
from .base import *
import logging
import tqdm

logger = logging.getLogger(__name__)

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


def M_step_alpha(*, alpha, gamma):
    
    N = gamma.shape[0]
    log_phat = log_dirichlet_expectation(gamma).mean(-2)
    
    return update_dir_prior(alpha, N, log_phat)


class LdaModel(BaseModel):

    n_contexts = 32

    def __init__(self, 
        seed = 0, 
        dtype = np.float32,
        pi_prior = 1.,
        m_prior = 1.,
        beta_prior = 1.,
        num_epochs = 10000, 
        difference_tol = 5e-3,
        estep_iterations = 100,
        bound_tol = 1e-2,
        n_components = 10,
        quiet = True):

        self.seed = seed
        self.difference_tol = difference_tol
        self.estep_iterations = estep_iterations
        self.num_epochs = num_epochs
        self.dtype = dtype
        self.bound_tol = bound_tol
        self.pi_prior = pi_prior
        self.m_prior = m_prior
        self.beta_prior = beta_prior
        self.n_components = n_components
        self.quiet = quiet


    def _bound(self, gamma, phi_matrix, weighted_phi):

        logE_gamma = log_dirichlet_expectation(gamma)

        evidence = 0
        evidence += np.sum(weighted_phi*(
                np.expand_dims(self.logE_epsilon, 0) + logE_gamma[:,:,None,None,None] \
                    + self.logE_lambda[None,:,:,:,None] - np.log(phi_matrix))
            )
        
        evidence += sum(dirichlet_bound(self.alpha, gamma, logE_gamma))
        
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
    

    def _inference(self, gamma, freq_matrix, difference_tol = 1e-2, iterations = 100):

        phi_matrix_prebuild = np.exp(self.logE_lambda)[:,:,:,None] * np.exp(self.logE_epsilon)    
        
        for i in range(iterations):

            old_gamma = gamma.copy()
            logE_gamma = log_dirichlet_expectation(gamma)

            phi_matrix_unnormalized = phi_matrix_prebuild * np.exp(logE_gamma)[:,:,None,None,None]
            phi_matrix = phi_matrix_unnormalized/phi_matrix_unnormalized.sum(1, keepdims = True)

            weighted_phi = phi_matrix*np.expand_dims(freq_matrix, 1)

            gamma = self.alpha + weighted_phi.sum(axis = (-1,-2,-3))

            if np.abs(gamma - old_gamma).mean() < difference_tol:
                logging.debug('Stopped E-step after {} iterations.'.format(str(i+1)))
                break

        return gamma, phi_matrix, weighted_phi


    def _update_model_parameters(self, weighted_phi):

        _lambda_sstats = weighted_phi.sum(axis = (0,-1))
        epsilon_sstats = weighted_phi.sum(axis = 0)
        
        epsilon = E_step_epsilon(
            epsilon_sstats = epsilon_sstats,
            b = self.b,
        )
        
        _lambda = E_step_lambda(nu = self.nu, 
                _lambda_sstats = _lambda_sstats)

        return epsilon, _lambda

        
    def _update_priors(self, gamma):

        b = M_step_b(b = self.b, epsilon = self.epsilon)

        alpha = M_step_alpha(alpha = self.alpha, gamma = gamma)

        nu = M_step_nu(nu = self.nu, _lambda = self._lambda)
    
        return b, alpha, nu
    

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

        self.bounds = []

        _it = range(self.num_epochs)
        if not self.quiet:
            _it = tqdm.tqdm(_it, desc = 'Training')

        for epoch in _it:
            
            gamma, phi_matrix, weighted_phi = \
                self._inference(
                    gamma, freq_matrix, 
                    difference_tol = self.difference_tol,
                    iterations = self.estep_iterations
                )

            self.epsilon, self._lambda = \
                self._update_model_parameters(weighted_phi)
            
            #if epoch > 10:
            self.b, self.alpha, self.nu = \
                self._update_priors(gamma)

            self.bounds.append(
                self._bound(gamma, phi_matrix, weighted_phi)
            )

            if epoch > 1 and (self.bounds[-1] - self.bounds[-2]) < self.bound_tol:
                break
    
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
                iterations = 1000
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
        )