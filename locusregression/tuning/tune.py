
from .hyperband import run_hyperband
from functools import partial
from ..model import LocusRegressor
from ..model import GBTRegressor
from ..model import logger
from ..corpus import train_test_split
import logging


def random_model(randomstate,
                min_components = 2, 
                max_components = 10,
                model_params = {},
                tune_subsample = True,
                locus_subsample_rates = [0.0625, 0.125, 0.25,],
                model_type = 'regression',
                ):
    
    if model_type == 'regression':
        basemodel = LocusRegressor
    elif model_type == 'gbt':
        basemodel = GBTRegressor
    else:
        raise ValueError(f'Unknown model type {model_type}')
    

    if tune_subsample:
        model_params['locus_subsample'] = randomstate.choice(locus_subsample_rates)
        model_params['batch_size'] = randomstate.choice([32,64,128])
        

    return basemodel(
            num_epochs=1000, 
            eval_every = 1000,
            n_components = randomstate.randint(min_components, max_components),
            **basemodel.sample_params(randomstate),
            **model_params
        )


def eval_params(model, resources,*, train, test):
    
    model.time_limit = resources
    model.partial_fit(train)
    return model.score(test)


def get_records(trial_num, bracket, model, loss, resources, model_type = 'regression'):
    
    record = {
            'bracket' : bracket,
            'param_n_components' : model.n_components,
            'param_locus_subsample' : model.locus_subsample,
            'param_seed' : int(model.seed),
            'param_tau' : int(model.tau),
            'param_kappa' : float(model.kappa),
            'param_batch_size' : int(model.batch_size),
            'resources' : resources,
            'score' : loss,
            'trial_num' : trial_num,
            'model_type' : model_type,
        }

    return record

    


def tune_model(corpus,
    n_jobs = 1, 
    seed = 0,
    train_size = 0.7,
    max_time = 100000000,
    factor = 4,
    successive_halving=False,*,
    tune_subsample = False,
    model_type = 'regression',
    locus_subsample_rates = [0.0625, 0.125, 0.25,],
    max_candidates = 150,
    max_brackets = 4,
    time_per_epoch = None,
    min_components, max_components,
    **model_params):
    
    assert(max_components>min_components)
    assert(n_jobs > 0)

    train, test = train_test_split(corpus, seed = seed,
                        train_size = train_size)

    max_candidates = 1000#(max_components - min_components)*3*(3 if tune_subsample else 1)

    logger.setLevel(logging.ERROR)
    #logging.basicConfig(level = logging.ERROR)

    if time_per_epoch is None:
        print('Estimating how long it will typically take to fit a model...')
        warmup_epochs = 10
        test_model = LocusRegressor(
            n_components=min_components + (max_components - min_components)//2,
            **model_params,
            num_epochs=warmup_epochs,
        ).fit(corpus)

        time_per_epoch = sum(test_model.elapsed_times)/warmup_epochs # the first iteration is usually twice as long as subsequent iterations
    
    
    max_time = min(max_time, int(time_per_epoch * 400))
    print(f'Allocating {max_time} seconds for each trial.')


    return run_hyperband(
            partial(random_model, 
                min_components = min_components, 
                max_components = max_components,
                model_params = model_params,
                tune_subsample = tune_subsample,
                model_type = model_type,
                locus_subsample_rates = locus_subsample_rates,
            ),
            partial(eval_params, train = train, test = test),
            partial(get_records, model_type = model_type),
            factor = factor,
            max_resources = max_time,
            successive_halving = successive_halving,
            seed = seed,
            max_candidates= max_candidates,
            n_jobs= n_jobs,
            max_brackets = max_brackets,
        )