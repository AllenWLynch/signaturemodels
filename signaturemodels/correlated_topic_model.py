import numpy as np
from .base import *
import logging
import tqdm
from scipy.optimize import minimize
from functools import partial
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from scipy.linalg import inv, det


def initialize_parameters( 
        m_prior = 1., 
        beta_prior = 1.,*,
        dtype, n_components, n_contexts):
    
    mu = np.zeros(n_components).astype(dtype)
    sigma = np.diag(np.ones(n_components)).astype(dtype)
    
    b = (np.ones((n_components,2,3)) * m_prior).astype(dtype, copy = False)
    
    nu = (np.ones((2, n_contexts//2)) * beta_prior).astype(dtype, copy = False)
    
    return b, nu, mu, sigma


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

    gamma = random_state.normal(0.,0.1, 
                               (n_samples, n_components) # n_genomes by n_components
                              ).astype(dtype, copy=False)

    v = np.square(random_state.gamma(100., 1./100., 
                               (n_samples, n_components) # n_genomes by n_components
                              ).astype(dtype, copy=False))
    
    return gamma, v

def M_step_mu_sigma(gamma, v):
    
    mu_hat = gamma.mean(0)
    sigma = 1/len(gamma) * ( (gamma - mu_hat).T.dot(gamma - mu_hat) \
        + np.array([np.diag(v[i]) for i in range(len(v))]).sum(0) )
    
    return mu_hat, sigma


def lambda_log_expectation(_lambda):
    
    n_comps, _, n_contexts = _lambda.shape
    flattended_expectation = log_dirichlet_expectation(
        _lambda.reshape((n_comps, 2*n_contexts))
    )
    
    return flattended_expectation.reshape((n_comps, 2, n_contexts))

def E_step_squigly(*,gamma, v):
    
    return np.sum(
        np.exp(gamma + v/2),
        axis = -1
    )

def E_step_gamma(*,weighted_phi, gamma, v, squigly,
                mu, sigma, freq):
    
    inv_sigma = inv(sigma)
    phisum = weighted_phi.sum(axis = (-3, -2, -1))
    
    def objective(gamma):

        val = -1/2 * (gamma - mu).dot(inv_sigma).dot((gamma - mu)[:,None])

        val += np.sum(weighted_phi*gamma[:,None,None,None]) + \
            freq*\
            ( 1 - np.log(squigly) - 1/squigly*np.sum(np.exp(gamma+v/2), axis = -1))
        
        return -val[0]  # maximize!!!!


    def jacobian(gamma):

        jac = np.squeeze(-np.dot(inv_sigma, (gamma - mu)[:,None]).T) + \
               phisum - (freq/squigly)*np.exp(gamma + v/2)

        return -jac # maximize!!!!

    initial_loss = -objective(gamma)
        
    new_gamma = minimize(
        objective, 
        gamma,
        jac = jacobian,
        method='bfgs',
    ).x
    
    improvement = -objective(new_gamma) - initial_loss

    return new_gamma, improvement


def E_step_v(*, gamma, v, squigly, freq,
                sigma):
    
    inv_sigma = inv(sigma)
    
    def objective(v_sq):
        
        val = -1/2 * np.trace(np.diag(v_sq).dot(inv_sigma)) \
            + freq*( 1 - np.log(squigly) - 1/squigly * np.sum(np.exp(gamma+v_sq/2), axis = -1)) \
            + 1/2 * np.sum(np.log(v_sq) + np.log(2*np.pi) + 1)
        
        return -val
    
    def jacobian(v_sq):
        
        jac = -np.diag(inv_sigma)/2 \
            - freq/(2*squigly) * np.exp(gamma + v_sq/2) \
            + 1/(2*v_sq)

        return -jac    
    
    initial_loss = -objective(v)
        
    new_v = minimize(
        objective, 
        v,
        jac = jacobian,
        method='l-bfgs-b',
        bounds= [(1e-10, np.inf) for _ in range(v.shape[-1])],
    ).x
    
    improvement = -objective(new_v) - initial_loss
        
    return new_v, improvement


class CorrelatedTopicModel(BaseModel):

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


    @property
    def sigma(self):
        return self._sigma

    @property
    def inv_sigma(self):
        return self._inv_sigma

    @sigma.setter
    def sigma(self, _sigma):
        self._sigma = _sigma
        self._inv_sigma = inv(_sigma)


    def _bound(self,*, gamma, v, phi_matrix, weighted_phi, freqs):

        logE_gamma = gamma # for consistency in bound equations

        evidence = 0
        evidence += np.sum(weighted_phi*(
                np.expand_dims(self.logE_epsilon, 0) + logE_gamma[:,:,None,None,None] \
                    + self.logE_lambda[None,:,:,:,None] - np.log(phi_matrix))
            )

        for i in range(len(freqs)):

            evidence += 1/2 * np.log(det(self.inv_sigma)) - self.n_components/2*np.log(2*np.pi) \
                 -1/2 * (np.trace(np.diag(v[i]).dot(self.inv_sigma)) + (gamma[i] - self.mu).dot(self.inv_sigma).dot((gamma[i] - self.mu)))

            evidence += -freqs[i]*np.log(np.sum(np.exp(gamma[i]+v[i]/2), axis = -1))

            evidence += 1/2 * np.sum(np.log(v[i]) + np.log(2*np.pi) + 1)
        
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


    def _inference(self,*, gamma, v, freq_matrix, freqs, difference_tol = 1e-2, iterations = 100,
                quiet = True):

        phi_matrix_prebuild = np.exp(self.logE_lambda)[:,:,:,None] * np.exp(self.logE_epsilon)

        phis, weighted_phis = [],[]

        _it = range(len(freq_matrix))
        if not quiet:
            _it = tqdm.tqdm(_it, desc = 'Infering latent variables')

        for i in _it:
    
            for j in range(iterations):

                old_gamma = gamma[i].copy()

                squigly = E_step_squigly(gamma = gamma[i], v = v[i])
                
                phi_matrix_unnormalized = phi_matrix_prebuild * np.exp(gamma[i,:,None,None,None])
                phi_matrix = phi_matrix_unnormalized/phi_matrix_unnormalized.sum(0, keepdims = True)
                
                weighted_phi = phi_matrix*np.expand_dims(freq_matrix[i], 0)

                gamma[i], _ = E_step_gamma(weighted_phi = weighted_phi,
                                            gamma = gamma[i], 
                                            v = v[i], 
                                            squigly = squigly,
                                            freq=freqs[i],
                                            mu = self.mu, sigma = self.sigma, 
                                        )
                
                v[i], _ = E_step_v(freq = freqs[i],
                                    gamma = gamma[i], 
                                    v = v[i], 
                                    squigly = squigly,
                                    sigma = self.sigma)


                mean_gamma_diff = np.abs(gamma[i] - old_gamma).mean()

                if mean_gamma_diff < difference_tol:
                    break
            
            phis.append(phi_matrix)
            weighted_phis.append(weighted_phi)

        return gamma, v, np.array(phis), np.array(weighted_phis)


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

        
    def _update_priors(self):

        b = M_step_b(b = self.b, epsilon = self.epsilon)

        nu = M_step_nu(nu = self.nu, _lambda = self._lambda)
    
        return b, nu


    def get_freqs(self, freq_matrix):
        return freq_matrix.reshape((len(freq_matrix), -1)).sum(-1)
    

    @extract_freqmatrix
    def fit(self, freq_matrix):

        assert isinstance(self.n_components, int) and self.n_components > 1

        self.random_state = np.random.RandomState(self.seed)

        self.b, self.nu, self.mu, self.sigma = initialize_parameters(
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

        gamma, v = initialize_document_variational_parameters(
            random_state = self.random_state, dtype = self.dtype, 
            n_components = self.n_components,
            n_samples= len(freq_matrix),
        )

        freqs = self.get_freqs(freq_matrix)
        self.bounds = []

        _it = range(self.num_epochs)
        if not self.quiet:
            _it = tqdm.tqdm(_it, desc = 'Training')

        for epoch in _it:

            logger.debug('Begin epoch {}.'.format(str(epoch)))
            
            gamma, v, phi_matrix, weighted_phi = \
                self._inference(
                    gamma = gamma, 
                    v = v, 
                    freq_matrix = freq_matrix, 
                    freqs = freqs,
                    difference_tol = self.difference_tol,
                    iterations = self.estep_iterations
                )

            self.epsilon, self._lambda = \
                self._update_model_parameters(weighted_phi)
            
            #if epoch > 10:
            self.b, self.nu = self._update_priors()

            self.mu, self.sigma = M_step_mu_sigma(gamma, v)
                    
            self.bounds.append(
                self._bound(gamma = gamma, v = v, phi_matrix = phi_matrix,
                    weighted_phi=weighted_phi, freqs=freqs)
            )

            if epoch > 1 and (self.bounds[-2] > self.bounds[-1]):
                logger.warning('Bounds did not decrease between epochs, training may be numerically unstable.\nThis usually occurs because the model has too many parameters for the training data.')

            if epoch > 25 and (self.bounds[-1] - self.bounds[-2]) < self.bound_tol:
                break
    
        return self

    @staticmethod
    def _monte_carlo_softmax_expecation(gamma, v, n_samples = 300):

        def softmax(x):
            return np.exp(x)/np.exp(x).sum(axis = -1, keepdims = True)

        z = np.random.randn(n_samples, *gamma.shape)

        return softmax(v[None, :, :]*z + gamma[None,:,:]).mean(axis = 0)



    def _infer_document_variables(self, freq_matrix):

        gamma, v = initialize_document_variational_parameters(
                    random_state=self.random_state,
                    dtype=self.dtype,
                    n_samples=len(freq_matrix),
                    n_components=self.n_components
                )

        freqs = self.get_freqs(freq_matrix)

        return self._inference(
                gamma = gamma, v = v,
                freq_matrix = freq_matrix,
                freqs = freqs,
                difference_tol = 5e-3, 
                iterations = 1000,
                quiet=False,
            )

    @extract_freqmatrix
    def predict(self, freq_matrix):

        gamma, v, _, _ = self._infer_document_variables(freq_matrix)
        return self._monte_carlo_softmax_expecation(gamma, v)
        

    @extract_freqmatrix
    def score(self, freq_matrix):

        gamma, v, phi_matrix, weighted_phi = \
            self._infer_document_variables(freq_matrix)
        
        freqs = self.get_freqs(freq_matrix)

        return self._bound(
            gamma = gamma, v = v, phi_matrix=phi_matrix,
            weighted_phi=weighted_phi, freqs = freqs,
        )
