from sklearn.ensemble._gb_losses import RegressionLossFunction, DummyRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import logging
from scipy.optimize import line_search

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
            - np.sum(y) * np.log( np.sum( sample_weight * np.exp(raw_predictions) ) )
        
        return -objective


    def negative_gradient(self, y, raw_predictions, **kargs):
        '''
        y - phis
        raw_predictions - f(X) = BX
        sample_weight - exposures
        '''

        exposure = kargs['sample_weight']
        y_hat = exposure*np.exp(raw_predictions)

        grads = y \
            - np.sum(y)/np.sum(y_hat) * y_hat

        return grads


    def hessian(self, y, raw_predictions, **kargs):
        
        exposure = kargs['sample_weight']
        y_hat = exposure*np.exp(raw_predictions)

        hess = -np.sum(y)/np.square(np.sum(y_hat)) * (y_hat * np.sum(y_hat) + y_hat**2)

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


def optim_tree(learning_rate = 1.,*,
        beta_sstats,
        prediction,
        X_matrices,
        window_sizes,
    ):

    beta_sstats = beta_sstats[0]
    X = X_matrices[0].T
    exposures = window_sizes[0].ravel()

    phis = np.array(list(beta_sstats.values()))
    indx = np.array(list(beta_sstats.keys()))

    y = np.zeros_like(prediction)
    y[indx] = phis

    loss = MutationRateLoss()
    
    first_order = loss.negative_gradient(y, prediction, 
                       sample_weight = exposures)

    second_order = loss.hessian(y, prediction, 
                       sample_weight = exposures) + 1e-30

    residuals = first_order/second_order

    tree = DecisionTreeRegressor(
        criterion='friedman_mse', 
        max_depth=3,
        min_samples_leaf=5,
        min_samples_split=5,
        ccp_alpha=0.01,
        ).fit(X, residuals)


    search_direction = tree.predict(X)

    alpha = line_search(
        lambda x : loss(y, x, sample_weight=exposures), 
        lambda x : -loss.negative_gradient(y, x,  sample_weight=exposures),
        prediction,
        search_direction
        )[0]

    if alpha is None:
        logger.info('Tree rejected')
        raise TreeFitError()

    return lambda x : alpha*learning_rate*tree.predict(x)