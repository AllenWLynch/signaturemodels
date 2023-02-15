from sklearn.ensemble._gb_losses import RegressionLossFunction, DummyRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import logging
logger = logging.getLogger('Trees')

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

        grads = y \
            - np.sum(y)/np.sum(exposure*np.exp(raw_predictions)) * exposure * np.exp(raw_predictions)

        return grads


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
    
    residuals = loss.negative_gradient(y, prediction, 
                       sample_weight = exposures)

    initial = loss(y, prediction, 
                       sample_weight = exposures)

    tree = DecisionTreeRegressor(
        criterion='friedman_mse', 
        max_depth=3,
        #ccp_alpha = 0.0001,
        ).fit(X, residuals)

    improvement = loss(y, prediction + learning_rate*tree.predict(X),
                    sample_weight=exposures)

    logger.debug(f'Improvement: {improvement - initial}')

    return tree