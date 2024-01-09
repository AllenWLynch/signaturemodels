from ..model import LocusRegressor
from ._gbt_modelstate import GBTModelState, GBTCorpusState
import locusregression.model.gbt._gbt_sstats as _gbt_sstats
import matplotlib.pyplot as plt


class GBTRegressor(LocusRegressor):

    MODEL_STATE = GBTModelState
    CORPUS_STATE = GBTCorpusState
    SSTATS = _gbt_sstats

    def __init__(self,
                 tree_learning_rate=0.1, 
                 max_depth = 5,
                 max_trees_per_iter = 25,
                 l2_regularization=0.,
                 n_iter_no_change=3,
                  **kw, 
                ):
        super().__init__(**kw)
        self.tree_learning_rate = tree_learning_rate
        self.max_depth = max_depth
        self.max_trees_per_iter = max_trees_per_iter
        self.l2_regularization=l2_regularization
        self.n_iter_no_change=n_iter_no_change

    
    def _get_rate_model_parameters(self):
        return {
            'tree_learning_rate' : self.tree_learning_rate,
            'max_depth' : self.max_depth, 
            'max_trees_per_iter' : self.max_trees_per_iter,
            'n_iter_no_change' : self.n_iter_no_change,
            'l2_regularization': self.l2_regularization,
        }

    
    @classmethod
    def sample_params(cls, trial):
        return dict(
            #tau = trial.suggest_categorical('tau', [1, 16, 48]),
            #n_iter_no_change = trial.suggest_categorical('n_iter_no_change', [1,3,5]),
            #l2_regularization = trial.suggest_categorical('l2_regularization', [0., 1e-10, 1e-7, 1e-5, 1e-3, 1e-2])
        )
    

    def plot_summary(self, fontsize = 7):
        
        _, ax = plt.subplots(
                            self.n_components,1,
                            figsize = (5.5, 1.25*self.n_components),
                            sharex = 'col',
                            )
        
        for i in range(self.n_components):

            self.plot_signature(i, ax = ax[i], normalization='global', fontsize=fontsize)
            ax[i].set(xlabel = '', ylabel = 'Component ' + str(i), title = '')
            ax[i].yaxis.label.set_size(fontsize)

        return ax
    

    def plot_compare_coefficients(self,*args,**kw):
        raise NotImplementedError(
            'GBTRegressor does not support plot_compare_coefficients'
        )
        

