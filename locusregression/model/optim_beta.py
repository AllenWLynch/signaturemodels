
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
            exposures, 
            X_matrix,
            beta_sstats,
        ):
        
        nonzero_indices = np.array(list(beta_sstats.keys()))
        weighted_phis = np.vstack(list(beta_sstats.values())).T # (K, l)
        
        K_weights = weighted_phis.sum(axis = -1, keepdims = True)
        deriv_second_term = weighted_phis @ X_matrix[:, nonzero_indices].T # K,L x L,F -> KxF

        normalizer = ( exposures * np.exp( beta_mu0 @ X_matrix + 1/2*(beta_nu0 @ X_matrix)**2 ) ).sum(-1, keepdims=True)

        gamma = K_weights/normalizer * exposures

        optim_kwargs = {
                'gamma' : gamma,
                'deriv_second_term' : deriv_second_term,
                'nonzero_indices' : nonzero_indices,
                'weighted_phis' : weighted_phis,
                'X_matrix' : X_matrix,
                }

        return partial(BetaOptimizer._objective_jac_sample, **optim_kwargs)


    @staticmethod
    def _objective_jac_regularization(beta_mu, beta_nu, tau):
        tau_sq = tau**2
        objective = -1/(2*tau_sq)*np.sum( np.square(beta_mu) + np.square(beta_nu) ) + np.sum(np.log(beta_nu))

        mu_jac = -1/tau_sq * beta_mu
        std_jac = -1/tau_sq * beta_nu + 1/beta_nu

        return -objective, -np.concatenate([mu_jac, std_jac])


    @staticmethod
    def _objective_jac_sample(
            beta_mu, 
            beta_nu,*,
            # suffstats calculated before optimization
            X_matrix,
            gamma,
            nonzero_indices,
            deriv_second_term,
            weighted_phis,
        ):

        locus_logmu = beta_mu @ X_matrix # F x F, L -> L
        std_inner = beta_nu @ X_matrix
        
        locus_expectation = gamma * np.exp(locus_logmu + 1/2*np.square(std_inner)) # L

        objective = np.sum(weighted_phis * locus_logmu[nonzero_indices]) - locus_expectation.sum(axis = -1)
               
        mu_jac = deriv_second_term - locus_expectation @ X_matrix.T 
        
        std_jac =  -(std_inner*locus_expectation) @ X_matrix.T
        
        return -objective, -np.squeeze(np.concatenate([mu_jac, std_jac], axis = 1))

    
    @staticmethod
    def optimize(
        tau = 1.,*, 
        beta_mu0, 
        beta_nu0,
        exposures,
        X_matrices,
        beta_sstats,
    ):
        return list(map(np.vstack, 
            zip(*[
                BetaOptimizer._optimize(
                    tau = tau[k], beta_mu0 = beta_mu0[k], beta_nu0 = beta_nu0[k],
                    exposures=exposures, X_matrices=X_matrices, 
                    beta_sstats=[{ l : v[k] for l,v in stats.items() } for stats in beta_sstats]
                )
                for k in range(beta_mu0.shape[0])
            ])
        ))
    

    @staticmethod
    def _optimize(tau = 1.,*, 
            beta_mu0, 
            beta_nu0,
            exposures,
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
                    exposures=_window,
                    X_matrix=_X,
                    beta_sstats=beta_sstats_sample,
                )
                for _window, _X, beta_sstats_sample in \
                    zip(exposures, X_matrices, beta_sstats) \
                    if len(beta_sstats_sample) > 0
            ]


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

        #beta_logger.debug('Update converged after {} iterations.'.format(optim_results.nfev))

        return new_beta[:F], new_beta[F:]