
import numpy as np
from scipy.optimize import minimize
import logging
from functools import partial
#from numba import njit
beta_logger = logging.getLogger('Beta optimizer')


#@njit
def _objective_jac_regularization(beta_mu, beta_nu, tau):
    tau_sq = tau**2
    objective = -1/(2*tau_sq)*np.sum( np.square(beta_mu) + np.square(beta_nu) ) + np.sum(np.log(beta_nu))

    mu_jac = -1/tau_sq * beta_mu
    std_jac = -1/tau_sq * beta_nu + 1/beta_nu

    dim = beta_mu.shape[0]
    jac = np.empty(dim*2, dtype=np.float64)
    jac[:dim] = mu_jac; jac[dim:] = std_jac

    return -objective, -jac #np.concatenate([mu_jac, std_jac])

    
#@njit
def _objective_jac_sample(
        beta_mu, 
        beta_nu,*,
        # suffstats calculated before optimization
        X_mut,
        X_negsample,
        gamma,
        deriv_second_term,
        weighted_phis,
    ):

    std_inner = beta_nu @ X_negsample
    locus_expectation = gamma * np.exp(beta_mu @ X_negsample + 1/2*np.square(std_inner)) # L

    objective = np.sum(weighted_phis * (beta_mu @ X_mut)[np.newaxis,:]) - locus_expectation.sum(axis = -1)
            
    mu_jac = deriv_second_term - locus_expectation @ X_negsample.T 
    
    std_jac =  -(std_inner*locus_expectation) @ X_negsample.T
    
    dim = beta_mu.shape[0]
    jac = np.empty(dim*2, dtype=np.float64)
    jac[:dim] = mu_jac; jac[dim:] = std_jac

    #print(objective)

    return -objective, -jac #np.concatenate([mu_jac, std_jac], axis = np.int64(1)).reshape(-1)



class BetaOptimizer:

    @staticmethod
    def _get_sample_optim_func(*,
            beta_mu0, 
            beta_nu0,
            exposures, 
            X_matrix,
            beta_sstats,
            negative_samples = None,
        ):
        
        _, n_loci = X_matrix.shape

        nonzero_indices = np.array(list(beta_sstats.keys()))
        weighted_phis = np.vstack(list(beta_sstats.values())).T # (K, l)
        
        K_weights = weighted_phis.sum(axis = -1, keepdims = True)

        X_mut = X_matrix[:, nonzero_indices].copy()

        deriv_second_term = weighted_phis @ X_mut.T # K,L x L,F -> KxF

        if not negative_samples is None:
            neg_sample_weight = n_loci/len(negative_samples)

            exposures = np.hstack([
                exposures[:, nonzero_indices], neg_sample_weight*exposures[:, negative_samples]
            ])

            #X_negsample = .copy()
            X_negsample = np.hstack([X_mut, X_matrix[:,negative_samples]])
        else:
            X_negsample = X_matrix.copy()

        normalizer = ( exposures * np.exp( beta_mu0 @ X_negsample + 1/2*(beta_nu0 @ X_negsample)**2 ) ).sum(-1, keepdims=True)

        gamma = K_weights/normalizer * exposures

        
        optim_kwargs = {
                'gamma' : gamma,
                'deriv_second_term' : deriv_second_term,
                'weighted_phis' : weighted_phis,
                'X_mut' : X_mut,
                'X_negsample' : X_negsample,
                }

        return partial(_objective_jac_sample, **optim_kwargs)


    @staticmethod
    def optimize(
        tau = 1.,
        negative_subsample = 10000,*, 
        beta_mu0, 
        beta_nu0,
        exposures,
        X_matrices,
        beta_sstats,
        random_state,
    ):
        
        if (not negative_subsample is None) and negative_subsample < X_matrices[0].shape[1]:
            negative_samples = random_state.choice(
                X_matrices[0].shape[1], 
                size = negative_subsample, 
                replace = False
            )
        else:
            negative_samples = None


        return list(map(np.vstack, 
            zip(*[
                BetaOptimizer._optimize(
                    tau = tau[k], beta_mu0 = beta_mu0[k], beta_nu0 = beta_nu0[k],
                    exposures=exposures, X_matrices=X_matrices, 
                    beta_sstats=[{ l : v[k] for l,v in stats.items() } for stats in beta_sstats],
                    negative_samples=negative_samples,
                )
                for k in range(beta_mu0.shape[0])
            ])
        ))
    

    @staticmethod
    def _optimize(
            tau = 1.,
            negative_samples = None,*, 
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
                    negative_samples=negative_samples,
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

            _obj_reg, _jac_reg = _objective_jac_regularization(beta_mu, beta_nu, tau)

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