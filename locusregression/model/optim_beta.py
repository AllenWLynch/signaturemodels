
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import logging
from functools import partial
beta_logger = logging.getLogger('Beta optimizer')


class BetaOptimizer:

    @staticmethod
    def _get_sample_optim_func(*,
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

        return partial(BetaOptimizer._objective_jac_sample, **optim_kwargs)#, \
                #partial(BetaOptimizer._hess_sample, **optim_kwargs)


    @staticmethod
    def _objective_jac_regularization(beta_mu, beta_nu, tau):
        tau_sq = tau**2
        objective = -1/(2*tau_sq)*np.sum( np.square(beta_mu) + np.square(beta_nu) ) + np.sum(np.log(beta_nu))

        mu_jac = -1/tau_sq * beta_mu
        std_jac = -1/tau_sq* beta_nu + 1/beta_nu

        return -objective, -np.concatenate([mu_jac, std_jac])


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
            K_weights,
        ):

        locus_logmu = np.dot(beta_mu, X_matrix) # F x F, L -> L
        std_inner = np.dot(beta_nu, X_matrix)
        
        locus_expectation = window_size * np.exp(locus_logmu + 1/2*np.square(std_inner)) # L
        
        log_denom = np.sum(locus_expectation, axis = -1, keepdims=True)/normalizer - 1 + np.log(normalizer) # (L,)
        
        objective = np.sum(weighted_phis * (np.log(window_size)[:,nonzero_indices] + locus_logmu[nonzero_indices] - log_denom))                    
               
        mu_jac = deriv_second_term - K_weights/normalizer * \
            np.dot(locus_expectation, X_matrix.T) 
        
        std_jac =  -K_weights/normalizer * np.dot(std_inner*locus_expectation, X_matrix.T)
        
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
    def optimize(tau = 1.,*, 
            beta_mu0, 
            beta_nu0,
            window_sizes,
            X_matrices,
            beta_sstats,
        ):

        F = len(beta_mu0)
        beta_0 = np.concatenate([beta_mu0, beta_nu0])

        '''
        Each sample is associated with its X_matrix, window_sizes/"marginal sensitivity", and beta suffstats.
        For each sample, calculate the sufficient statistics for the gradient update,
        Then return functions which can be called with args (beta_mu, beta_nu) to calculate the objective score,
        jacobian, and hessians. - the suffstats are already provided as arguments to that function.        
        '''
        obj_jac_funcs = [
                BetaOptimizer._get_sample_optim_func(
                    beta_mu0=beta_mu0,
                    beta_nu0=beta_nu0,
                    window_size=_window,
                    X_matrix=_X,
                    beta_sstats=beta_sstats_sample,
                )
                for _window, _X, beta_sstats_sample in \
                    zip(window_sizes, X_matrices, beta_sstats)
            ]

            #hess = lambda x : hess_func(x[:F], x[F:])

        def reduce_objective_jac(x):
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

            _obj_reg, _jac_reg = BetaOptimizer._objective_jac_regularization(beta_mu, beta_nu, tau)

            obj+=_obj_reg
            jac+=_jac_reg

            return obj, jac

        '''
        def hess(x):

            raise NotImplementedError()
            
            beta_mu, beta_nu = x[:F], x[F:]

            hess_matrix = 0

            for hess_sample in hess_funcs:
                hess_matrix += hess_sample(beta_mu, beta_nu)

            return hess_matrix
        
        hess_kwargs = dict(
            hess = hess,
            method = 'trust-constr',
            constraints=LinearConstraint(
                np.hstack([np.zeros((F,F)), np.diag(np.ones(F))]),
                np.zeros(F),
                np.inf*np.ones(F),
                keep_feasible=True,
            ),
            options = dict(
                xtol = 1e-5,
                gtol = 1e-5,
            )
        )'''

        lbfgs_kwargs = dict(
            method = 'l-bfgs-b',
            bounds = [(-np.inf, np.inf)]*F + [(1e-30,np.inf)]*F
        )

        optim_results = minimize(
            reduce_objective_jac, 
            beta_0,
            jac = True,
            **lbfgs_kwargs,
        )

        new_beta = optim_results.x

        beta_logger.debug('Update converged after {} iterations.'.format(optim_results.nfev))


        return new_beta[:F], new_beta[F:]