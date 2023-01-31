
import numpy as np
from scipy.special import psi, gammaln, polygamma
import logging
logger = logging.getLogger(__name__)


def log_dirichlet_expectation(alpha):
    
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(np.sum(alpha))
    else:
        return psi(alpha) - psi(np.sum(alpha, axis = -1, keepdims=True))


def dirichlet_bound(alpha, gamma, logE_gamma):

    alpha = np.expand_dims(alpha, 0)
    
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
        tol = 1e-8, max_iterations = 1000):
        
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
        logger.debug('Prior update did not converge.')
        return initial_prior
    
    return prior


def update_tau(mu, nu):
    return np.sqrt(2*np.pi) * np.sum(mu**2 + nu**2, axis = -1)


def M_step_alpha(alpha, gamma):
    
    N = gamma.shape[0]
    log_phat = log_dirichlet_expectation(gamma).mean(-2)
    
    return update_dir_prior(alpha, N, log_phat, rho = 0.1)