import shap
import logging
from numpy.random import RandomState
from numpy import vstack, squeeze
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
        return squeeze(
            tree_explainer.shap_values(
                chunk,
                check_additivity=False,
                approximate=False,
            )
        )
    
    component = model._get_signature_idx(signature)

    if not model.is_trained:
        logger.warn('This model was not trained to completion, results may be innaccurate')
    
    tree_model = model.model_state.rate_models[component]

    corpus_state = model.corpus_states[corpus.name].clone_corpusstate(corpus)
    
    X_tild = model.model_state.feature_transformer\
        .transform({corpus.name : corpus_state})

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

    num_chunks = corpus.shape[1] // chunk_size + 1

    shap_values = vstack(
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_calculate_shap_values)(explainer, X_tild[i*chunk_size:(i+1)*chunk_size])
            for i in range(num_chunks)
        )
    )

    return shap_values