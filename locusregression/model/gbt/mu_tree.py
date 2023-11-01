from sklearn.ensemble._gb_losses import RegressionLossFunction, DummyRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import logging
from scipy.optimize import line_search
from functools import partial
logger = logging.getLogger('Trees')

class TreeFitError(ValueError):
    pass

class MutationRateLoss(RegressionLossFunction):

    def init_estimator(self):
        return DummyRegressor(strategy="mean")

    def __call__(self, y, raw_predictions, sample_weight=None):
        '''
        y - phis
        raw_predictions - f(X) = BX
        sample_weight - exposures
        '''

        objective = np.dot(y, raw_predictions) \
            - np.sum(y) * np.log( np.sum( sample_weight * np.exp(raw_predictions) ) ) \
            - 1/2*np.sum(raw_predictions**2)
        
        return -objective


    def negative_gradient(self, y, raw_predictions, **kargs):
        '''
        y - phis
        raw_predictions - f(X) = BX
        sample_weight - exposures
        '''

        exposure = kargs['sample_weight']
        y_hat = exposure*np.exp(raw_predictions)

        grads = y - np.sum(y)/np.sum(y_hat) * y_hat - raw_predictions

        return grads


    def hessian(self, y, raw_predictions, **kargs):
        
        exposure = kargs['sample_weight']
        y_hat = exposure*np.exp(raw_predictions)

        hess = -np.sum(y)/np.square(np.sum(y_hat)) * (y_hat * np.sum(y_hat) + y_hat**2) - 1

        return -hess


    def update_terminal_regions(
        self,
        tree,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
        sample_mask,
        learning_rate=0.1,
        k=0,
    ):
        pass
    

    def _update_terminal_region(
        self,
        tree,
        terminal_regions,
        leaf,
        X,
        y,
        residual,
        raw_predictions,
        sample_weight,
    ):
        """LAD updates terminal regions to median estimates."""
        pass



def _tree_predict(x,*,alpha, learning_rate, tree):
    return alpha*learning_rate*0.1*tree.predict(x)


def optim_tree(learning_rate = 1.,*,
        beta_sstats,
        logmus, 
        lognus,
        X_matrices,
        exposures,
    ):
    
    X = np.hstack(X_matrices)
    exposures = np.concatenate([expos * np.exp(1/2*(lognu**2)) 
                           for expos, lognu in zip(exposures, lognus)
                         ], axis = 1)
            
    logmu0 = np.concatenate(logmus)

    phis = [phi for corpus_beta in beta_sstats for phi in corpus_beta.values()]
    indx = [idx + X_matrices[0].shape[1]*i 
            for i, corpus_beta in enumerate(beta_sstats)
            for idx in corpus_beta.keys()
            ]

    y = np.zeros_like(exposures)
    y[:,indx] = phis

    loss = MutationRateLoss()

    X = X.T
    y = np.squeeze(y)

    first_order = loss.negative_gradient(y, logmu0, 
                       sample_weight = exposures)

    second_order = loss.hessian(y, logmu0, 
                       sample_weight = exposures)

    residuals = np.squeeze(first_order/second_order)

    tree = DecisionTreeRegressor(
        criterion='friedman_mse', 
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=5,
        ccp_alpha=0.01,
        ).fit(X, residuals)

    search_direction = tree.predict(X)

    alpha = line_search(
        lambda x : loss(y, x, sample_weight=exposures), 
        lambda x : -loss.negative_gradient(y, x,  sample_weight=exposures),
        logmu0,
        search_direction
        )[0]

    if alpha is None:
        logger.info('Tree rejected')
        raise TreeFitError()


    return partial(_tree_predict,
                alpha = alpha,
                learning_rate = learning_rate,
                tree = tree    
            )