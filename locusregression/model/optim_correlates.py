
import numpy as np
from scipy.optimize import LinearConstraint, minimize, Bounds
from scipy.special import psi, digamma, polygamma, loggamma
import logging
from functools import partial
beta_logger = logging.getLogger('Beta optimizer')
delta_logger = logging.getLogger('Beta optimizer')


def M_step_delta(*,delta_sstats, beta_sstats, delta, trinuc_distributions):
    
    nonzero_indices = np.array(list(beta_sstats.keys()))
    weighted_phis = np.vstack(list(beta_sstats.values())).T # (K, l)
    
    trinuc_distributions = trinuc_distributions[:, nonzero_indices]
    
    marginal_sstats = weighted_phis.sum()
    
    
    def objective_jac(delta):
        
        lamsum = delta.sum()
        E_loglambda = psi(delta) - psi(lamsum)
        
        val = np.dot(delta_sstats, E_loglambda) \
            - np.dot(weighted_phis, np.log(np.dot(delta, trinuc_distributions)) - np.log(lamsum))
        
        val -= loggamma(lamsum) - loggamma(delta).sum() + np.dot(delta - 1, E_loglambda)
        
        zero_term = (delta_sstats - delta + 1)
        
        j = polygamma(1, delta)*zero_term - polygamma(1, lamsum)*np.sum(zero_term) \
            - np.squeeze(
                np.dot(trinuc_distributions, (weighted_phis*1/np.dot(delta, trinuc_distributions)).T) # (w) * ((m)x(m,w) -> w
             ) + marginal_sstats/lamsum
            
        return -val, -j
    
    
    def hess(delta):
        
        lamsum = delta.sum()
        zero_term = (delta_sstats - delta + 1)
        
        constants = -polygamma(2, lamsum)*np.sum(zero_term) - marginal_sstats/np.square(lamsum)
        
        square = np.dot(trinuc_distributions, # (m,w) x (w) --> (m)
                   (weighted_phis * 1/np.square(np.dot(delta, trinuc_distributions)) * trinuc_distributions).T
                  )
        
        diag = np.diag(polygamma(2, delta)*zero_term)
        
        hess_matrix = square + diag + constants
        
        return -hess_matrix
    
    
    #initial_loss = -objective_jac(delta_sstats+1)[0]
    
    newdelta = minimize(
        objective_jac, 
        delta_sstats+1,
        method = 'tnc',
        jac = True,
        bounds = Bounds(1e-30, np.inf, keep_feasible=True),
    ).x
    
    #improvement = -objective_jac(newdelta)[0] - initial_loss
    
    return newdelta


class BetaOptimizer:

    @staticmethod
    def _get_optim_func(*,
            beta_mu0, 
            beta_nu0,
            window_size, 
            X_matrix,
            beta_sstats,
        ):
        
        nonzero_indices = np.array(list(beta_sstats.keys()))
        weighted_phis = np.vstack(list(beta_sstats.values())).T # (K, l)
        
        K_weights = weighted_phis.sum(axis = -1, keepdims = True)
        deriv_second_term = np.dot(weighted_phis, X_matrix[:, nonzero_indices].T) # K,L x L,F -> KxF

        locus_std_sq = np.square(np.dot(beta_nu0, X_matrix))
        normalizer = ( window_size * np.exp(np.dot(beta_mu0, X_matrix) + 1/2*locus_std_sq) ).sum(-1, keepdims=True)

        optim_kwargs = {
                'K_weights' : K_weights, 
                'normalizer' : normalizer, 
                'deriv_second_term' : deriv_second_term,
                'nonzero_indices' : nonzero_indices,
                'weighted_phis' : weighted_phis,
                'window_size' : window_size,
                'X_matrix' : X_matrix,
                }

        return partial(BetaOptimizer._objective_jac_sample, **optim_kwargs), \
                partial(BetaOptimizer._hess_sample, **optim_kwargs)


    @staticmethod
    def _objective_jac_sample(
            beta_mu, 
            beta_nu,*,
            # suffstats calculated before optimization
            window_size,
            X_matrix,
            normalizer,
            weighted_phis,
            nonzero_indices,
            deriv_second_term,
            K_weights
        ):

        locus_logmu = np.dot(beta_mu, X_matrix) # F x F, L -> L
        std_inner = np.dot(beta_nu, X_matrix)
        
        locus_expectation = window_size * np.exp(locus_logmu + 1/2*np.square(std_inner)) # L
        
        log_denom = np.sum(locus_expectation, axis = -1, keepdims=True)/normalizer - 1 + np.log(normalizer) # (L,)
        
        objective = -1/2*np.sum(np.square(beta_mu)  + np.square(beta_nu)) \
            + np.sum(weighted_phis * (np.log(window_size)[:,nonzero_indices] + locus_logmu[nonzero_indices] - log_denom)) \
            + np.sum(np.log(beta_nu)) # scalar
               
        mu_jac = -beta_mu + deriv_second_term - K_weights/normalizer * \
            np.dot(locus_expectation, X_matrix.T) # (L) x (L,F) -> (F)
        
        std_jac = -beta_nu -K_weights/normalizer * \
            np.dot(std_inner*locus_expectation, X_matrix.T) + 1/beta_nu

        print(objective)
        
        return -objective, -np.squeeze(np.concatenate([mu_jac, std_jac], axis = 1))
    

    @staticmethod
    def _hess_sample(
            beta_mu, 
            beta_nu,*,
            window_size,
            X_matrix,
            # suffstats calculated before optimization
            normalizer,
            weighted_phis,
            nonzero_indices,
            deriv_second_term,
            K_weights
        ):

        F = beta_mu.size

        std_inner = np.dot(beta_nu, X_matrix)
        locus_expectation = window_size * np.exp(np.dot(beta_mu, X_matrix) + 1/2*np.square(std_inner))*X_matrix # L,F
        
        dmu_dmu = -np.diag(np.ones(F)) - K_weights/normalizer*np.dot(locus_expectation, X_matrix.T) # (F,F) + (F,L)x(L,F) --> (F,F)
        
        dmu_dstd = -K_weights/normalizer*np.dot(locus_expectation * std_inner, X_matrix.T)
        
        dstd_dstd = -np.diag(np.ones(F)) - K_weights/normalizer*np.dot(locus_expectation * (np.square(std_inner) + 1), X_matrix.T) \
            - np.diag(1/np.square(beta_nu))
        
        hess_matrix = np.vstack([
            np.hstack([dmu_dmu, dmu_dstd]), 
            np.hstack([dmu_dstd.T, dstd_dstd])
        ])
        
        return -hess_matrix

    
    @staticmethod
    def optimize(*, 
            beta_mu0, 
            beta_nu0,
            window_size, 
            X_matrix,
            beta_sstats,
            shared_correlates,
        ):

        F = len(beta_mu0)
        beta_0 = np.concatenate([beta_mu0, beta_nu0])

        if shared_correlates:
            '''
            Each sample is associated with its X_matrix, window_sizes/"marginal sensitivity", and beta suffstats.
            For each sample, calculate the sufficient statistics for the gradient and hessian update,
            Then return functions which can be called with args (beta_mu, beta_nu) to calculate the objective score,
            jacobian, and hessians. - the suffstats are already provided as arguments to that function.        
            '''
            obj_jac_funcs, hess_funcs = list(zip(*\
                [
                    BetaOptimizer._get_optim_func(
                        beta_mu0=beta_mu0,
                        beta_nu0=beta_nu0,
                        window_size=window_size_sample,
                        X_matrix=X_matrix_sample,
                        beta_sstats=beta_sstats_sample
                    )
                    for window_size_sample, X_matrix_sample, beta_sstats_sample in\
                        zip(window_size,  X_matrix, beta_sstats)
                ]
            ))


            def objective_jac(x):
                '''
                The objective and gradient are summed over each training example
                for a new estimate for beta_mu/nu - "x".
                '''
                
                beta_mu, beta_nu = x[:F], x[F:]

                obj = 0
                jac = np.zeros(2*F)

                for obj_jac_sample in obj_jac_funcs:
                    _obj, _jac = obj_jac_sample(beta_mu, beta_nu)

                    obj+=_obj
                    jac+=_jac

                return obj, jac


            def hess(x):
                '''
                The objective and gradient are summed over each training example
                for a new estimate for beta_mu/nu - "x".
                '''
                
                beta_mu, beta_nu = x[:F], x[F:]

                hess_matrix = 0

                for hess_sample in hess_funcs:
                    hess_matrix += hess_sample(beta_mu, beta_nu)

                return hess_matrix

        else:

            objective_jac, hess = BetaOptimizer._get_optim_func(
                        beta_mu0=beta_mu0,
                        beta_nu0=beta_nu0,
                        window_size=window_size,
                        X_matrix=X_matrix,
                        beta_sstats=beta_sstats
                    )

        optim_results = minimize(
            objective_jac, 
            beta_0,
            jac = True,
            hess = hess,
            method = 'trust-constr',
            constraints=LinearConstraint(
                np.hstack([np.zeros((F,F)), np.diag(np.ones(F))]),
                1e-10*np.zeros(F),
                np.inf*np.ones(F),
                keep_feasible=True,
            ),
            options={'xtol' : 1e-5}
        )

        new_beta = optim_results.x

        beta_logger.debug('Update converged after {} iterations.'.format(optim_results.nit))

        return new_beta





def M_step_mu_nu(*,beta_mu, window_size,
                    beta_nu, X_matrix,
                    beta_sstats):
    # beta_sstats -> a dictionary with 
    # X_matrix -> (F x L)
    
    locus_std_sq = np.square(np.dot(beta_nu, X_matrix))
    normalizer = ( window_size * np.exp(np.dot(beta_mu, X_matrix) + 1/2*locus_std_sq) ).sum(-1, keepdims=True)
    
    nonzero_indices = np.array(list(beta_sstats.keys()))
    weighted_phis = np.vstack(list(beta_sstats.values())).T # (K, l)
    
    K_weights = weighted_phis.sum(axis = -1, keepdims = True)
    deriv_second_term = np.dot(weighted_phis, X_matrix[:, nonzero_indices].T) # K,L x L,F -> KxF
    
    F = len(beta_mu)
    
    def objective_jac(x):
        
        # beta_mu (K x F)
        
        beta_mu, beta_nu = x[:F], x[F:]
        
        locus_logmu = np.dot(beta_mu, X_matrix) # F x F, L -> L
        std_inner = np.dot(beta_nu, X_matrix)
        locus_expectation = window_size * np.exp(locus_logmu + 1/2*np.square(std_inner)) # L
        
        log_denom = np.sum(locus_expectation, axis = -1, keepdims=True)/normalizer - 1 + np.log(normalizer) # (L,)
        
        objective = -1/2*np.sum(np.square(beta_mu)  + np.square(beta_nu)) \
            + np.sum(weighted_phis * (np.log(window_size)[:,nonzero_indices] + locus_logmu[nonzero_indices] - log_denom)) \
            + np.sum(np.log(beta_nu)) # scalar
               
        mu_jac = -beta_mu + deriv_second_term - K_weights/normalizer * \
            np.dot(locus_expectation, X_matrix.T) # (L) x (L,F) -> (F)
        
        std_jac = -beta_nu -K_weights/normalizer * \
            np.dot(std_inner*locus_expectation, X_matrix.T) + 1/beta_nu
        
        return -objective, -np.squeeze(np.concatenate([mu_jac, std_jac], axis = 1))
    
    
    def hess(x):
        
        beta_mu, beta_nu = x[:F], x[F:]
        
        std_inner = np.dot(beta_nu, X_matrix)
        locus_expectation = window_size * np.exp(np.dot(beta_mu, X_matrix) + 1/2*np.square(std_inner))*X_matrix # L,F
        
        dmu_dmu = -np.diag(np.ones(F)) - K_weights/normalizer*np.dot(locus_expectation, X_matrix.T) # (F,F) + (F,L)x(L,F) --> (F,F)
        
        dmu_dstd = -K_weights/normalizer*np.dot(locus_expectation * std_inner, X_matrix.T)
        
        dstd_dstd = -np.diag(np.ones(F)) - K_weights/normalizer*np.dot(locus_expectation * (np.square(std_inner) + 1), X_matrix.T) \
            - np.diag(1/np.square(beta_nu))
        
        hess_matrix = np.vstack([
            np.hstack([dmu_dmu, dmu_dstd]), 
            np.hstack([dmu_dstd.T, dstd_dstd])
        ])
        
        return -hess_matrix
    
    
    initial = np.concatenate([beta_mu, beta_nu])
    
    #initial_loss = -objective_jac(initial)[0]
    
    new = minimize(
        objective_jac, 
        initial,
        method = 'trust-constr',
        constraints=LinearConstraint(
            np.hstack([np.zeros((F,F)), np.diag(np.ones(F))]),
            1e-10*np.zeros(F),
            np.inf*np.ones(F),
            keep_feasible=True,
        ),
        jac = True,
        hess = hess,
        options={'xtol' : 1e-5}
    ).x
    
    #print(-objective_jac(new)[0] - initial_loss)
    
    return new[:F], new[F:]