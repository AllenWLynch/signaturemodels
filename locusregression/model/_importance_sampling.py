import numpy as np
import tqdm
from functools import partial
from scipy.special import logsumexp
from ._dirichlet_update import log_dirichlet_expectation
from scipy.stats import dirichlet_multinomial


def _conditional_logp_mutation_locus(*, model_state, sample, corpus_state):
        '''
        Calculates an array of log \Tilde{P}_{z|m,l}(z|m,l) for each z in {1,...,K}
        These probabilities are unnormalized, and are used in the gibbs sampling step
        '''

        _lambda = model_state.delta/np.sum(model_state.delta, axis = -1, keepdims = True)
        rho = model_state.omega/np.sum(model_state.omega, axis = -1, keepdims = True)

        mutation_matrix = _lambda[:,:,None] * rho

        log_p_m_l = np.log(corpus_state.trinuc_distributions[sample.context, sample.locus]) \
                    + np.log(mutation_matrix[:, sample.context, sample.mutation]) \
                    - np.log( np.dot(_lambda, corpus_state.trinuc_distributions[:,sample.locus]) ) # K,C x C,L -> K,I
        
        #log_mutrates = np.log(sample.exposures) + beta @ corpus_state.X_matrix
        log_p_l = np.log(sample.exposures[:, sample.locus]) + corpus_state.logmu[:, sample.locus] - corpus_state.log_denom(sample.exposures)

        return log_p_l + log_p_m_l # K,I



def _model_logp_given_z(log_p_ml_z, z, alpha):
    
    K, N = log_p_ml_z.shape
    N_z = np.array([np.sum(z == k) for k in range(K)])

    return log_p_ml_z[z, np.arange(N)] + 1/N*( dirichlet_multinomial.logpmf(N_z, alpha, N) )


def _categorical_draw(p, randomstate):
    assert np.isclose(p.sum(0) - 1, 0).all() 
    K,N = p.shape

    draw = randomstate.uniform(0,1,N)
    return np.argmax(p.cumsum(0) > draw, axis = 0)


def _gibbs_sample(z, temperature = 1,*,
                  alpha, log_p_ml_z, N, K, randomstate):


    N_z = np.array([np.sum(z == k) for k in range(K)])[:,np.newaxis]
    
    log_q_z = temperature*log_p_ml_z + np.log( N_z + alpha ) - np.log( N - 1 + alpha )

    q_z = np.exp( log_q_z  - logsumexp(log_q_z, axis = 0, keepdims = True) )

    z = _categorical_draw(q_z, randomstate)

    return z


def _get_gibbs_sample_function(log_p_ml_z,*,alpha, randomstate = None):

    if randomstate is None:
        randomstate = np.random.RandomState(0)

    K, N = log_p_ml_z.shape
    
    q_z = np.repeat(alpha[:,np.newaxis]/alpha.sum(), N, axis = 1)

    z = _categorical_draw(q_z, randomstate)
    
    return partial(
            _gibbs_sample,
            alpha = alpha[:,np.newaxis], 
            log_p_ml_z = log_p_ml_z, 
            K= K,
            N = N, 
            randomstate = randomstate
        ), z


def _annealed_importance_sampling(
    log_p_ml_z,*,alpha,
    n_iters = 100, n_samples_per_iter = 100,
):
    
    temperatures = np.linspace(0,1,n_samples_per_iter)

    weights = []

    for i in tqdm.tqdm(range(n_iters + 1), ncols=100, desc = 'Importance sampling iterations'):
         
        gibbs_sample, z_tild = _get_gibbs_sample_function(
             log_p_ml_z, 
             alpha = alpha, 
             randomstate = np.random.RandomState(i)
        )

        iter_weights_running = _model_logp_given_z(log_p_ml_z, z_tild, alpha) * temperatures[0]

        for j in range(1,n_samples_per_iter):
            
            z_tild = gibbs_sample(z_tild, temperature = temperatures[j])

            iter_weights_running = iter_weights_running + _model_logp_given_z(log_p_ml_z, z_tild, alpha) * (temperatures[j] - temperatures[j-1])

        weights.append(iter_weights_running)

    return weights



def _get_z_posterior(log_p_ml_z,*,alpha, 
                     n_iters = 1000,
                     randomstate = None):
    
    gibbs_sampler, z_tild = _get_gibbs_sample_function(log_p_ml_z, 
                                               alpha = alpha, 
                                               randomstate = randomstate)
    
    K, N = log_p_ml_z.shape
    z_posterior = np.zeros_like(log_p_ml_z)

    for step in tqdm.tqdm(range(1,n_iters), ncols=100, desc = 'Sampling mutation assignments'):

        z_tild = gibbs_sampler(z_tild, temperature= 1.)
        
        z_posterior[z_tild, np.arange(N)] += 1

    return z_posterior / np.sum(z_posterior, axis = 0, keepdims = True)
    