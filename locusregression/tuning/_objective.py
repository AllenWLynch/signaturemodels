
from ..model import LocusRegressor
from ..model import GBTRegressor
from ..model import _pseudo_r2
import optuna
from tqdm import trange
import sys


def objective(trial,
            min_components = 2, 
            max_components = 10,
            model_params = {},
            tune_subsample = True,
            locus_subsample_rates = [0.125, 0.25, None],
            model_type = 'regression',
            subset_by_loci=True,
            no_improvement=5,*,
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
        sample_params['batch_size'] = trial.suggest_categorical('batch_size', [64,128,None])

    model_params.update(sample_params)

    print(
        'Training model with params:\n' + \
        '\n'.join(
            [f'\t{k} = {v}' for k,v in model_params.items()]
        ) + '\n',
        file = sys.stderr,
    )
    
    model = basemodel(
            eval_every = 5,
            quiet=False,
            seed = trial.number,
            num_epochs=num_epochs,
            **model_params,
        )
    
    scores = []

    train_empirical = train.get_empirical_mutation_rate(); train_null = train.context_frequencies
    test_empirical = test.get_empirical_mutation_rate(); test_null = test.context_frequencies

    for train_score, test_score in model._fit(
        train, test_corpus=test, subset_by_loci=subset_by_loci,
    ):
        step = model.epochs_trained
        trial.report(test_score, step)
        scores.append(test_score)

        trial.set_user_attr(f'test_score_{step}',  test_score)
        trial.set_user_attr(f'train_score_{step}',  train_score)
        
        test_mutation_r2 = _pseudo_r2(test_empirical, np.exp(model.get_log_marginal_mutation_rate(test)), test_null)
        train_mutation_r2 = _pseudo_r2(train_empirical, np.exp(model.get_log_marginal_mutation_rate(train)), train_null)
                
        trial.set_user_attr(f'test_mutation_r2_{step}', test_mutation_r2)
        trial.set_user_attr(f'train_mutation_r2_{step}', train_mutation_r2)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if len(scores) > no_improvement and not (min(scores) in scores[-no_improvement:]):
            break
    
    trial.set_user_attr('test_mutation_r2', model.get_mutation_rate_r2(test))
    trial.set_user_attr('train_mutation_r2', model.get_mutation_rate_r2(train))

    return model.score(test, subset_by_loci=subset_by_loci) 
