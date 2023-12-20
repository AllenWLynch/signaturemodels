
from ..model import LocusRegressor
from ..model import GBTRegressor
import optuna
from tqdm import trange

def objective(trial,
            min_components = 2, 
            max_components = 10,
            model_params = {},
            tune_subsample = True,
            locus_subsample_rates = [0.0625, 0.125, 0.25,],
            model_type = 'regression',*,
            num_epochs,
            train, test,
            ):
    
    
    if model_type == 'regression':
        basemodel = LocusRegressor
    elif model_type == 'gbt':
        basemodel = GBTRegressor
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    sample_params = basemodel.sample_params(trial)
    sample_params['n_components'] = trial.suggest_int('n_components', min_components, max_components)

    if tune_subsample:
        sample_params['locus_subsample'] = trial.suggest_categorical('locus_subsample', locus_subsample_rates)
        sample_params['batch_size'] = trial.suggest_categorical('batch_size', [16,32,64,1000000])

    model_params.update(sample_params)
    
    model = basemodel(
            eval_every = 1000000,
            quiet=True,
            **model_params,
            seed = trial.number,
            num_epochs=1,
        )
    
    for i in trange(1, num_epochs + 1, desc = 'Training', ncols=100, 
                        position=0, leave=True):
        
        model.num_epochs = i
        model._fit(train, reinit = False)

        if i % 25 == 0:
            intermediate_score = model.score(test)
            trial.report(intermediate_score, i)

            if trial.should_prune():
                raise optuna.TrialPruned()
            
    
    return model.score(test)

