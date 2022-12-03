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


def E_step_squigly(*,gamma, v):
    
    return np.sum(
        np.exp(gamma + v/2),
        axis = -1
    )

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


def E_step_gamma(*,weighted_phi, gamma, v, squigly,
                mu, sigma, freqs):
    
    inv_sigma = inv(sigma)
    phisum = weighted_phi.sum(axis = (-3, -2, -1))
    
    def objective(curr_gamma, i):

        v_sq = v[i]

        val = -1/2 * (curr_gamma - mu).dot(inv_sigma).dot((curr_gamma - mu)[:,None])

        val += np.sum(weighted_phi[i]*curr_gamma[:,None,None,None]) + \
            freqs[i]*\
            ( 1 - np.log(squigly[i]) - 1/squigly[i]*np.sum(np.exp(curr_gamma+v_sq/2), axis = -1))
        
        return -val[0]  # maximize!!!!


    def jacobian(curr_gamma, i):

        jac = np.squeeze(-np.dot(inv_sigma, (curr_gamma - mu)[:,None]).T) + \
               phisum[i] - (freqs[i]/squigly[i])*np.exp(curr_gamma + v[i]/2)

        return -jac # maximize!!!!
    
    new_gamma = np.zeros_like(gamma)
    
    improvement = []
    for i in range(len(freqs)):
        
        initial_loss = -objective(gamma[i], i)
        
        new_gamma[i] = minimize(
            partial(objective, i=i), 
            gamma[i],
            jac = partial(jacobian, i=i),
            method='bfgs',
        ).x
        
        improvement.append(-objective(new_gamma[i], i) - initial_loss)

    return new_gamma, sum(improvement)


def E_step_v(*, gamma, v, squigly, freqs,
                sigma):
    
    inv_sigma = inv(sigma)
    
    def objective(v_sq, i):
        
        val = -1/2 * np.trace(np.diag(v_sq).dot(inv_sigma)) \
            + freqs[i]*( 1 - np.log(squigly[i]) - 1/squigly[i] * np.sum(np.exp(gamma[i]+v_sq/2), axis = -1)) \
            + 1/2 * np.sum(np.log(v_sq) + np.log(2*np.pi) + 1)
        
        return -val
    
    def jaccobian(v_sq, i):
        
        jac = -np.diag(inv_sigma)/2 \
            - freqs[i]/(2*squigly[i]) * np.exp(gamma[i] + v_sq/2) \
            + 1/(2*v_sq)

        return -jac    
    
    new_v = np.ones_like(v)
        
    improvement = []
    for i in range(len(freqs)):
        
        initial_loss = -objective(v[i], i)
        
        new_v[i] = minimize(
            partial(objective, i=i), 
            v[i], 
            jac = partial(jaccobian, i=i),
            bounds = [(1e-8, np.inf) for _ in range(v.shape[-1])],
            method = 'l-bfgs-b',
        ).x
        
        improvement.append(-objective(new_v[i], i) - initial_loss)
        
    return new_v, sum(improvement)


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

        if not quiet:
            min_ = np.log(difference_tol)

        max_ = None
    
        for i in range(iterations):

            old_gamma = gamma.copy()

            squigly = E_step_squigly(gamma = gamma, v = v)
            
            phi_matrix_unnormalized = phi_matrix_prebuild * np.exp(gamma[:,:,None,None,None])
            phi_matrix = phi_matrix_unnormalized/phi_matrix_unnormalized.sum(1, keepdims = True)
            
            weighted_phi = phi_matrix*np.expand_dims(freq_matrix, 1)

            gamma, _ = E_step_gamma(weighted_phi = weighted_phi,
                                        gamma = gamma, v = v, squigly = squigly,
                                        mu = self.mu, sigma = self.sigma, freqs=freqs)
            
            v, _ = E_step_v(freqs = freqs,
                                gamma = gamma, v = v, squigly = squigly,
                                sigma = self.sigma)

            mean_gamma_diff = np.abs(gamma - old_gamma).mean()

            if not quiet:
                if max_ is None:
                    max_ = np.log(mean_gamma_diff)
                    range_ = max_ - min_
                    convergence_bar = tqdm.tqdm(total = 100, desc = 'Convergence')
                
                progress = int( 100 * (1-(max(np.log(mean_gamma_diff) - min_, 0)/range_)) )
                convergence_bar.update(max(progress - convergence_bar.n, 0))

            if  mean_gamma_diff < difference_tol:
                logging.debug('Stopped E-step after {} iterations.'.format(str(i+1)))
                break

        return gamma, v, phi_matrix, weighted_phi


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
