
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, ShuffleSplit
import numpy as np
import logging
from math import ceil
logger = logging.getLogger('Tuner')
logger.setLevel(logging.INFO)


def tune_model(model, corpus,
    n_jobs = 1, 
    seed = 0,
    seed_reps=1, 
    max_candidates = 50,
    cv = 3,
    tune_pi_prior = False,
    num_tournaments = 4,
    target_epochs = 300,
    min_epochs = 20,
    factor = 3,*,
    min_components, max_components):
    
    assert(seed_reps>=1)
    assert(max_components>min_components)
    assert(n_jobs > 0)

    param_grid = {
            'n_components' : list(range(min_components, max_components + 1)),
            'seed' : [2 + i for i in range(seed_reps)]
        }    

    if tune_pi_prior:
        param_grid['pi_prior'] = [0.5,1.,5.]

    if cv == 1:
        cv = ShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)

    #possible_combinations = np.prod([len(k) for k in param_grid.values()])
    #num_initial_candidates = min(possible_combinations, max_candidates)
    #num_tournaments = min( num_tournaments, int(np.log(num_initial_candidates/n_jobs)/np.log(factor)) + 1 )
    #logger.info('Running {} tournaments for tuning.'.format(num_tournaments))

    min_epochs = max(min_epochs, int( target_epochs*(1/factor)**(num_tournaments-1) ) )
    
    reglogger = logging.getLogger("LocusRegressor")
    init_level = logger.level
    reglogger.setLevel(logging.ERROR)

    try:
        grid = HalvingRandomSearchCV(
            model,
            param_grid,
            n_jobs=n_jobs, 
            verbose = 10, 
            cv = cv, 
            refit = False,
            min_resources= min_epochs,
            factor = factor,
            max_resources= target_epochs,
            n_candidates = max_candidates,
            resource='num_epochs',
            random_state=seed,
            )\
            .fit(corpus)
    finally:
        reglogger.setLevel(init_level)

    return grid