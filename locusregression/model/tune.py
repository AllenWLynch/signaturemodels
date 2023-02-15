
from .hyperband import run_hyperband
from functools import partial
from .model import LocusRegressor
from .model import logger
from ..corpus import train_test_split
import logging


def random_model(randomstate,
                    min_components = 2, 
                    max_components = 10,
                    model_params = {},
                    tune_subsample = True,
                    ):

    if tune_subsample:
        model_params['locus_subsample'] = randomstate.choice([0.0625, 0.125, 0.25])
        
    return LocusRegressor(
            num_epochs=0, 
            n_components = randomstate.randint(min_components, max_components),
            seed = randomstate.randint(0, 100000000),
            tau = randomstate.choice([1, 5, 25, 50, 100]),
            kappa = randomstate.choice([0.5, 0.6, 0.7]),
            **model_params
        )


def eval_params(model, resources,*, train, test):
    
    model.num_epochs = resources
    model.partial_fit(train)
    return model.score(test)


def get_records(bracket, model, loss, resources):
    
    return {
            'bracket' : bracket,
            'param_n_components' : model.n_components,
            'param_locus_subsample' : model.locus_subsample,
            'param_seed' : model.seed,
            'resources' : resources,
            'score' : loss,
        }


def tune_model(corpus,
    n_jobs = 1, 
    seed = 0,
    train_size = 0.7,
    max_epochs = 300,
    factor = 3,
    successive_halving=True,*,
    tune_subsample = False,
    min_components, max_components,
    **model_params):
    
    assert(max_components>min_components)
    assert(n_jobs > 0)

    train, test = train_test_split(corpus, seed = seed,
                        train_size = train_size)

    max_candidates = 1000#(max_components - min_components)*3*(3 if tune_subsample else 1)

    #logging.basicConfig(level = logging.ERROR)
    logger.setLevel(logging.ERROR)

    return run_hyperband(
            partial(random_model, 
                min_components = min_components, 
                max_components = max_components,
                model_params = model_params,
                tune_subsample = tune_subsample,
            ),
            partial(eval_params, train = train, test = test),
            get_records,
            factor = factor,
            max_resources = max_epochs,
            seed = seed,
            max_candidates= max_candidates,
            n_jobs= n_jobs,
        )