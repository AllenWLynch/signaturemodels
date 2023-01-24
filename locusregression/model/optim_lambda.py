import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import psi, digamma, polygamma, loggamma
import logging
from functools import partial
lambda_logger = logging.getLogger('Lambda optimizer')

class LambdaOptimizer:

    @staticmethod
    def _get_sample_optim_func(*,beta_sstats, trinuc_distributions):
        
        nonzero_indices = np.array(list(beta_sstats.keys()))
        weighted_phis = np.vstack(list(beta_sstats.values())).T # (K, l)
        
        trinuc_distributions = trinuc_distributions[:, nonzero_indices]

        optim_kwargs = {
            'trinuc_distributions' : trinuc_distributions,
            'weighted_phis' : weighted_phis,
        }

        return partial(LambdaOptimizer._objective_jac_sample, **optim_kwargs)


    @staticmethod
    def _objective_jac_regularization(delta, delta_sstats):
        
        lamsum = delta.sum()
        marginal_sstats = delta_sstats.sum()

        objective = np.dot(delta_sstats, psi(delta)-psi(lamsum)) + marginal_sstats*np.log(lamsum)
        objective -= loggamma(lamsum) - loggamma(delta).sum() + np.dot(delta - 1, psi(delta) - psi(lamsum))

        zero_term = (delta_sstats - delta + 1)
        jac = polygamma(1, delta)*zero_term - polygamma(1, lamsum)*np.sum(zero_term) + marginal_sstats/lamsum

        return -objective, -jac


    @staticmethod
    def _objective_jac_sample(delta,*,
         weighted_phis, 
         trinuc_distributions,
        ):
        
        objective = -np.dot(weighted_phis, np.log(np.dot(delta, trinuc_distributions)) )
        
        jac = -np.squeeze(
                np.dot(trinuc_distributions, (weighted_phis*1/np.dot(delta, trinuc_distributions)).T) # (w) * ((m)x(m,w) -> w
             )
            
        return -objective, -jac


    @staticmethod
    def _hess_sample():
        pass

    
    @staticmethod
    def optimize(delta0,*,
        trinuc_distributions,
        beta_sstats,
        delta_sstats
        ):
        
        '''
        Each sample is associated with its X_matrix, and beta suffstats.
        For each sample, calculate the sufficient statistics for the gradient and hessian update,
        Then return functions which can be called with args (beta_mu, beta_nu) to calculate the objective score,
        jacobian, and hessians. - the suffstats are already provided as arguments to that function.        
            '''

        obj_jac_funcs = [
                LambdaOptimizer._get_sample_optim_func(
                    trinuc_distributions = trinuc,
                    beta_sstats = beta_sstats_sample, 
                )
                for trinuc, beta_sstats_sample in \
                    zip(trinuc_distributions, beta_sstats)
            ]


        def reduce_objective_jac(delta):
            '''
            The objective and gradient are summed over each training example
            for a new estimate for beta_mu/nu - "x".
            '''
            obj = 0
            jac = np.zeros_like(delta)

            for obj_jac_sample in obj_jac_funcs:
                _obj, _jac = obj_jac_sample(delta)

                obj+=_obj
                jac+=_jac

            _obj_reg, _jac_reg = LambdaOptimizer._objective_jac_regularization(delta, delta_sstats)

            obj+=_obj_reg
            jac+=_jac_reg

            return obj, jac


        optim_results = minimize(
                reduce_objective_jac, 
                (delta0 + delta_sstats + 1)/2,
                method = 'l-bfgs-b',
                jac = True,
                bounds = Bounds(1e-30, np.inf, keep_feasible=True),
            )

        new_delta = optim_results.x

        lambda_logger.debug('Update converged after {} iterations.'.format(optim_results.nfev))

        return new_delta

