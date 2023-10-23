import numpy as np
from scipy.spatial.distance import cdist

def signature_cosine_distance(model, simulation_parameters):
    
    sigs = np.vstack([
        model.signature(i, raw = True, normalization='global') for i in range(model.n_components)
    ])

    truth = simulation_parameters['signatures'].reshape(-1,96)

    cosine_matches = 1 - cdist(sigs, truth, metric='cosine')
    
    return cosine_matches.max(0).mean()



def coef_l1_distance(model, simulation_parameters):
    
    return cdist(model.model_state.beta_mu[:,:-1], 
            simulation_parameters['beta'], 
            metric='cityblock'
            ).min(0).mean()
