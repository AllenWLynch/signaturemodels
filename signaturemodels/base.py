import numpy as np
from scipy.special import psi, gammaln, polygamma
import re
from functools import wraps as functoolswraps
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import pickle
import logging
logger = logging.getLogger(__name__)

MUTATION_MATCH = re.compile('([ATGC])\[([TC])\>([ATGC])\]([ATGC])')

CONTEXT_MAP = dict(zip([left+right for left in 'ATGC' for right in 'ATGC'], range(4**2)))
CATEGORY_MAP = {'C' : 0, 'T' : 1}
TRANSITION_MAP = {'C' : {'A' : 0, 'G' : 1, 'T' : 2},
                        'T' : {'A' : 0, 'C' : 1, 'G' : 2}
                }

def unravel_freqs():
    for cat_ in CATEGORY_MAP.keys():
        for context in CONTEXT_MAP.keys():
            for transition in TRANSITION_MAP[cat_].keys():
                yield '{left}[{initial}>{transition}]{right}'.format(
                    left = context[0],
                    initial = cat_,
                    transition = transition,
                    right = context[1],
                )

POSSIBLE_MUTATIONS = list(unravel_freqs())

mutation_pallete = {
    ('C','A') : 'cyan',
    ('C','G') : 'black',
    ('C','T') : 'red',
    ('T','A') : 'lightgrey',
    ('T','C') : 'lime',
    ('T','G') : 'pink'
}

def sort_mutations(mutation):
    left, initial, transition, right = re.findall(MUTATION_MATCH, mutation)[0]
    return initial, transition

SORTED_MUTATIONS = sorted(POSSIBLE_MUTATIONS, key=sort_mutations)

MUTATION_PALETTE = []
for mutation in SORTED_MUTATIONS:
    left, initial, transition, right = re.findall(MUTATION_MATCH, mutation)[0]
    MUTATION_PALETTE.append(
        mutation_pallete[(initial,transition)]
    )


def log_dirichlet_expectation(alpha):
    
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(np.sum(alpha))
    else:
        return psi(alpha) - psi(np.sum(alpha, axis = -1, keepdims=True))

    
def dirichlet_bound(alpha, gamma, logE_gamma):

    alpha = alpha[None, :]
    
    return gammaln(np.sum(alpha)) - gammaln(np.sum(gamma, axis = -1)) + \
        np.sum(
             gammaln(gamma) - gammaln(alpha) + (alpha - gamma)*logE_gamma,
             axis = -1
        )

def _dir_prior_update_step(prior, N, logphat, rho = 0.05):
    
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    return rho * dprior + prior


def update_dir_prior(prior, N, logphat, rho = 0.05,
    optimize = True, tol = 1e-8, max_iterations = 1000):
        
    failures = 0
    initial_prior = prior.copy()

    for it_ in range(max_iterations): #max iterations

        old_prior = prior.copy()
        prior = _dir_prior_update_step(prior, N, logphat, rho = rho)

        if not np.all(prior > 0) and np.all(np.isfinite(prior)):
            if failures > 5:
                logger.debug('Prior update failed at iteration {}. Reverting to old prior'.format(str(it_)))
                return initial_prior
            else:
                prior = old_prior
                rho*=1/2
                failures+=1

        elif np.abs(old_prior-prior).mean() < tol:
            break
    else:
        logger.info('Prior update did not converge.')
        return initial_prior
    
    return prior

'''def update_dir_prior(prior, N, logphat, rho = 0.05, optimize = True):

    rho = 0.1 if optimize else rho

    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    new_prior = rho * dprior + prior

    if not (np.all(prior > 0) and np.all(np.isfinite(prior))):
        logger.debug('Prior update failed. Reverting to old prior')
        return prior

    else:
        return new_prior'''


def E_step_epsilon(*,b, epsilon_sstats):
    return np.expand_dims(b, -2) + epsilon_sstats


def E_step_lambda(*,nu, _lambda_sstats):
    return np.expand_dims(nu, 0) + _lambda_sstats


def M_step_b(*, b, epsilon, rho, optimize):
    
    new_b = np.zeros_like(b)
    
    for k in range(b.shape[0]):
        for cat in [0,1]:
            
            log_phat = log_dirichlet_expectation(epsilon[k,cat]).mean(-2)
            
            new_b[k, cat] = update_dir_prior(
                            b[k, cat], 
                            epsilon.shape[-2],
                            log_phat,
                            rho = rho,
                            optimize = optimize,
                        )
    return new_b


def M_step_nu(*, nu, _lambda, rho, optimize):
    
    N = _lambda.shape[0]
    _lambda = _lambda.reshape(N,-1)
    log_phat = log_dirichlet_expectation(_lambda).mean(0)
    

    return update_dir_prior(
            nu.reshape(-1), N, log_phat, 
            rho = rho, optimize=optimize
            ).reshape(2,-1)


def lambda_log_expectation(_lambda):
    
    n_comps, _, n_contexts = _lambda.shape
    flattended_expectation = log_dirichlet_expectation(
        _lambda.reshape((n_comps, 2*n_contexts))
    )
    
    return flattended_expectation.reshape((n_comps, 2, n_contexts))


def convert_signature_to_freqtensor(**mutation_freqs):

    mutations = list(mutation_freqs.keys())
    freqs = list(mutation_freqs.values())

    assert all([isinstance(m, str) and re.match(MUTATION_MATCH, m) for m in mutations])

    freqtensor = np.zeros((2, len(CONTEXT_MAP), 3))

    for mutation, freq in zip(mutations, freqs):

        left, initial, transition, right = re.findall(MUTATION_MATCH, mutation)[0]
        freqtensor\
            [CATEGORY_MAP[initial]]\
            [CONTEXT_MAP[left+right]]\
            [TRANSITION_MAP[initial][transition]] += freq

    return freqtensor


def extract_freqmatrix(func):

    @functoolswraps(func)
    def convert_to_tensor(model, mutation_matrix):

        return func(model, 
            np.array([
                convert_signature_to_freqtensor(
                    **dict(zip(row.index, row.values))
                )
                for i, row in mutation_matrix.iterrows()
            ])
        )

    return convert_to_tensor
        

class BaseModel(BaseEstimator):

    def __init__(self, 
        seed = 0, 
        tau = 1.,
        kappa = 0.5,
        batch_size = 256,
        eval_every = 10,
        dtype = np.float32,
        pi_prior = 0.1,
        m_prior = 0.1,
        beta_prior = 0.1,
        num_epochs = 10000, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        n_components = 10,
        prior_update_every = 5,
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
        self.tau = tau
        self.kappa = kappa
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.prior_update_every = prior_update_every

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # epsilon setters and getters
    # whenever epsilon is updated, track the log expectation
    @property
    def epsilon(self):
        return self._epsilon

    @property
    def logE_epsilon(self):
        return self._logE_epsilon

    @epsilon.setter
    def epsilon(self, _epsilon):
        self._epsilon = _epsilon
        self._logE_epsilon = log_dirichlet_expectation(_epsilon)

    # lambda setters and getter
    @property
    def _lambda(self):
        return self._lambda_

    @property
    def logE_lambda(self):
        return self._logE_lambda

    @_lambda.setter
    def _lambda(self, _lambda):
        self._lambda_ = _lambda
        self._logE_lambda = lambda_log_expectation(_lambda)


    @staticmethod
    def get_global_update_scale(corpus_size, sstats_size):
        return corpus_size/sstats_size

    @staticmethod
    def svi_update(old, new, rho):
        return (1-rho)*old + rho*new


    def get_rho(self, epoch):
        return (self.tau + epoch)**-self.kappa


    def _estimate_global_parameters(self, weighted_phi, rho, sstat_scale = 1.):

        _lambda_sstats = weighted_phi.sum(axis = (0,-1)) * sstat_scale # scale to whole dataset size
        epsilon_sstats = weighted_phi.sum(axis = 0) * sstat_scale
        
        epsilon_hat = E_step_epsilon(
            epsilon_sstats = epsilon_sstats,
            b = self.b,
        )
        
        _lambda_hat = E_step_lambda(
            nu = self.nu, 
            _lambda_sstats = _lambda_sstats
            )

        return self.svi_update(self.epsilon, epsilon_hat, rho), \
            self.svi_update(self._lambda, _lambda_hat, rho)


    def _estimate_global_priors(self, rho, optimize = True):

        b_hat = M_step_b(b = self.b, epsilon = self.epsilon, 
                rho = rho, optimize = optimize)

        nu_hat = M_step_nu(nu = self.nu, _lambda = self._lambda, 
            rho = rho, optimize = optimize)
    
        return self.svi_update(self.b, b_hat, rho), \
            self.svi_update(self.nu, nu_hat, rho)
    

    def signature_posterior(self, signature):

        assert isinstance(signature, int) and signature >= 0 and signature < self.n_components

        context  = np.exp(lambda_log_expectation(self._lambda))[:,:,:,None]
        transition = np.exp(log_dirichlet_expectation(self.epsilon))

        score_dict = dict(zip(
            POSSIBLE_MUTATIONS, (context * transition).reshape(-1, len(POSSIBLE_MUTATIONS))[signature]
        ))

        return {mut : score_dict[mut] for mut in SORTED_MUTATIONS}


    def plot_signature(self, signature, figsize=(10,2), ax = None):

        sig = self.signature_posterior(signature)

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize= figsize)

        ax.bar(
            height = list(sig.values()),
            x = SORTED_MUTATIONS,
            width = 1,
            edgecolor = 'black',
            linewidth = 0.3,
            color = MUTATION_PALETTE,
        )

        ax.bar(
            height = list(sig.values()),
            x = SORTED_MUTATIONS,
            width = 1,
            color = MUTATION_PALETTE,
        )

        for s in ['left','right','top']:
            ax.spines[s].set_visible(False)
            
        ax.set(yticks = [], xticks = [])

        return ax