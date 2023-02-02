import numpy as np
import logging
from math import ceil
logger = logging.getLogger('Tuner')
logger.setLevel(logging.INFO)
from math import log, floor, ceil
import tqdm
from joblib import Parallel, delayed
from functools import partial
from locusregression.model.model import LocusRegressor


def random_models(n_configs, randomstate,
                  min_components = 2, 
                  max_components = 10,
                  model_params = {}
                  ):
    
    for i in range(n_configs):
        yield LocusRegressor(
            num_epochs=0, 
            n_components = randomstate.randint(min_components, max_components),
            seed = randomstate.randint(0, 100000000),
            **model_params
        )


def eval_params(model, resources,*, train, test):
    
    model.num_epochs = resources
    model.partial_fit(train)
    return model.score(test), model


def dry_run(model, resources):
    
    final_score = -(model.n_components - 5)**2 + 5
    process_score = log(resources + 1)
    
    return final_score*process_score + np.random.randn(), model


def hyperband(random_model_func, trial_func, 
              factor = 3,
              max_resources = 300,
              seed = 0,
              n_jobs = 1,
              successive_halving = False
            ):
    
    randomstate = np.random.RandomState(seed)
    
    R = max_resources
    s_max = min( floor( log(R)/log(factor) ), 4)
    B = (s_max + 1)*R
    
    results = []
    
    try:
        for s in range(s_max, 0, -1) if not successive_halving else [3]:
            
            n = ceil(B/R * (factor**s)/(s+1))
            
            models = list(random_model_func(n, randomstate))
            n = min(n, len(models))

            r = R*factor**(-s)
            
            print(f'Starting bracket {s} with {n} configurations')
                    
            for i in range(s+1):
                
                n_i = floor(n*factor**(-i))
                r_i = ceil(r*factor**i)
                
                print(f'\tStarting rung {i} with {len(models)} remaining models')
                
                losses, models = list(zip(*\
                    Parallel(n_jobs = n_jobs, verbose = 0)\
                    (delayed(trial_func)(model, r_i) 
                    for model in tqdm.tqdm(models, desc='\tStarting trials', ncols = 100)
                    )
                ))
                
                top_n = max(floor(n_i/factor), n_jobs)
                
                for loss, model in zip(losses, models):
                    results.append((model.n_components, model.seed, loss, r_i, s))
                
                ranks = np.argsort(-np.array(losses)).argsort()
                models = [model for rank, model in zip(ranks, models) if rank < top_n]
                
                if len(models) == 0:
                    break
    
    except KeyboardInterrupt as err:
        if len(results) > 0:
            pass
        else:
            raise err

    return dict(zip(['n_components', 'seed','loss','resources','bracket'] , list(zip(*results))))


def tune_model(corpus,
    n_jobs = 1, 
    seed = 0,
    train_size = 0.7,
    max_epochs = 300,
    factor = 3,
    successive_halving=False,*,
    min_components, max_components,
    **model_params):
    
    assert(max_components>min_components)
    assert(n_jobs > 0)

    train, test = corpus.split_train_test(seed, train_size = train_size)

    model_gen = partial(random_models,
        min_components=min_components,
        max_components=max_components,
        model_params = model_params
    )

    eval_fn = partial(
        eval_params, 
        train = train, 
        test = test,
    )

    grid = hyperband(
        model_gen, eval_fn, 
        factor = factor,
        max_resources = max_epochs,
        seed = seed,
        successive_halving = successive_halving,
        n_jobs = n_jobs,
    )

    return grid