import numpy as np
import tqdm
from functools import partial
from scipy.special import logsumexp

def _conditional_logp_mutation_locus(*, model_state, sample, corpus_state):
        '''
        Calculates an array of log \Tilde{P}_{z|m,l}(z|m,l) for each z in {1,...,K}
        These probabilities are unnormalized, and are used in the gibbs sampling step
        '''

        _lambda = model_state.delta/np.sum(model_state.delta, axis = 1, keepdims = True)
        rho = model_state.omega/np.sum(model_state.omega, axis = 1, keepdims = True)

        mutation_matrix = _lambda[:,:,None] * rho

        log_p_m_l = np.log(corpus_state.trinuc_distributions[sample.context, sample.locus]) \
                    + np.log(mutation_matrix[:, sample.context, sample.mutation]) \
                    - np.log( np.dot(_lambda, corpus_state.trinuc_distributions[:,sample.locus]) ) # K,C x C,L -> K,I
        
        #log_mutrates = np.log(sample.exposures) + beta @ corpus_state.X_matrix
        log_p_l = np.log(sample.exposures[:, sample.locus]) + corpus_state.logmu[:, sample.locus] - corpus_state.log_denom(sample.exposures)

        return log_p_l + log_p_m_l # K,I


def _model_logp_given_z(log_p_ml_z, z):
    n_obs = log_p_ml_z.shape[1]
    return np.sum(log_p_ml_z[z, np.arange(n_obs)])


def _gibbs_sample(temperature = 1,*,
                  N_z, z, alpha, log_p_ml_z, N, randomstate):

    for i in range(N):

        N_z[z[i]] = N_z[z[i]] - 1
        
        log_q_z = temperature * log_p_ml_z[:, i] + np.log( N_z + alpha ) - np.log( N - 1 + alpha )

        z[i] = np.argmax(log_q_z + randomstate.gumbel(size=log_q_z.shape))

        N_z[z[i]] = N_z[z[i]] + 1

    return z


def _get_gibbs_sample_function(log_p_ml_z,*,alpha, randomstate = None):

    if randomstate is None:
        randomstate = np.random.RandomState(0)

    K, N = log_p_ml_z.shape

    pi = randomstate.dirichlet(alpha, size = N)
    z = np.array([
         randomstate.choice(K, p = pi[i]) for i in range(N)
        ])
    
    N_z = np.array([np.sum(z == k) for k in range(K)])
    
    return partial(
            _gibbs_sample,
            N_z = N_z, z = z, 
            alpha = alpha, 
            log_p_ml_z = log_p_ml_z, 
            N = N, 
            randomstate = randomstate
        ), z


def _annealed_importance_sampling(
    log_p_ml_z,*,alpha,
    n_iters = 100, n_samples_per_iter = 100,
):
    
    temperatures = np.linspace(0,1,n_samples_per_iter)

    weights = []

    for i in tqdm.tqdm(range(n_iters), ncols=100, desc = 'Importance sampling iterations'):
         
        gibbs_sample, z_tild = _get_gibbs_sample_function(
             log_p_ml_z, 
             alpha = alpha, 
             randomstate = np.random.RandomState(i)
        )

        iter_weights_running = _model_logp_given_z(log_p_ml_z, z_tild) * temperatures[1]

        for j in range(2,n_samples_per_iter):
            
            z_tild = gibbs_sample(temperature = temperatures[j])

            iter_weights_running += _model_logp_given_z(log_p_ml_z, z_tild) * (temperatures[j] - temperatures[j-1])

        weights.append(iter_weights_running)

    return logsumexp(weights) - np.log(n_iters)



def _get_z_posterior(log_p_ml_z,*,alpha, 
                     n_iters = 1000, 
                     warm_up_steps = 25,
                     randomstate = None):
    
    gibbs_sampler, _ = _get_gibbs_sample_function(log_p_ml_z, 
                                               alpha = alpha, 
                                               randomstate = randomstate)
    
    _, N = log_p_ml_z.shape
    z_posterior = np.zeros_like(log_p_ml_z)

    for step in range(1,n_iters):

        z_tild = gibbs_sampler(temperature= min(1, step/warm_up_steps))

        if step > warm_up_steps:
            z_posterior[z_tild, np.arange(N)] += 1

    return z_posterior / np.sum(z_posterior, axis = 0, keepdims = True)
    