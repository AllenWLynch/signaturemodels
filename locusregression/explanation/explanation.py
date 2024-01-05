import shap
import logging
from numpy.random import RandomState
from numpy import vstack
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

def explain(
    signature,*,
    model,
    corpus,
    n_jobs = 1,
    chunk_size = 10000,
):
    
    def _calculate_shap_values(tree_explainer, chunk):
        return tree_explainer.shap_values(
            chunk,
            check_additivity=False,
            approximate=False,
        )
    
    component = model._get_signature_idx(signature)

    if not model.is_trained:
        logger.warn('This model was not trained to completion, results may be innaccurate')
    
    tree_model = model.model_state.rate_models[component]

    X_tild = corpus.X_matrix.T  

    background_idx = RandomState(0)\
                        .choice(
                            len(X_tild), 
                            size = 500, 
                            replace = False
                        )

    explainer = shap.TreeExplainer(
        tree_model,
        X_tild[background_idx],
        check_additivity=False
    )

    num_chunks = len(corpus.X_matrix.T) // chunk_size

    shap_values = vstack([
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_calculate_shap_values)(explainer, X_tild[i*chunk_size:(i+1)*chunk_size])
            for i in range(num_chunks)
        )
    ])

    return shap_values