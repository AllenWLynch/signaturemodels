from ..model import LocusRegressor
from ._gbt_modelstate import GBTModelState, GBTCorpusState
import locusregression.model.gbt._gbt_sstats as _gbt_sstats
import matplotlib.pyplot as plt


class GBTRegressor(LocusRegressor):

    MODEL_STATE = GBTModelState
    CORPUS_STATE = GBTCorpusState
    SSTATS = _gbt_sstats

    
    @classmethod
    def sample_params(cls, trial):
        return dict(
            tau = trial.suggest_categorical('tau', [1, 1, 1, 16, 48, 128]),
            kappa = trial.suggest_categorical('kappa', [0.5, 0.5, 0.5, 0.6, 0.7]),
            tree_learning_rate = trial.suggest_categorical('tree_learning_rate', [0.1, 0.25, 0.5, 1.]),
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
        

