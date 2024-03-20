
from ..model import LocusRegressor
from ..model import GBTRegressor
from ..model import _pseudo_r2
import optuna
from tqdm import trange
import sys
from numpy import exp
from functools import partial

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
    
    
    if model_type == 'linear':
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

    eval_r2_test = [partial(_pseudo_r2, 
                                y = subcorpus.get_empirical_mutation_rate(),
                                y_null=test.context_frequencies
                            )
                        for subcorpus in test.corpuses
                    ]

    eval_r2_train = [partial(_pseudo_r2, 
                                y = subcorpus.get_empirical_mutation_rate(),
                                y_null=train.context_frequencies
                            )
                        for subcorpus in train.corpuses
                    ]
    
    for train_score, test_score in model._fit(
        train, test_corpus=test, subset_by_loci=subset_by_loci,
    ):
        step = model.epochs_trained
        trial.report(test_score, step)
        scores.append(test_score)

        trial.set_user_attr(f'test_score_{step}',  test_score)
        trial.set_user_attr(f'train_score_{step}',  train_score)
        
        for train_, test_ in zip(train.corpuses, test.corpuses):
            trial.set_user_attr(f'test_mutationR2_{step}_{test_.name}', eval_r2_test(y_hat=exp(model.get_log_marginal_mutation_rate(test_))))
            trial.set_user_attr(f'train_mutationR2_{step}_{train_.name}', eval_r2_test(y_hat=exp(model.get_log_marginal_mutation_rate(train_))))

        if trial.should_prune():
            raise optuna.TrialPruned()

        if len(scores) > no_improvement and not (min(scores) in scores[-no_improvement:]):
            break
    
    for train_, test_ in zip(train.corpuses, test.corpuses):
        trial.set_user_attr(f'test_mutationR2_{test_.name}', eval_r2_test(y_hat=exp(model.get_log_marginal_mutation_rate(test_))))
        trial.set_user_attr(f'train_mutationR2_{train_.name}', eval_r2_test(y_hat=exp(model.get_log_marginal_mutation_rate(train_))))

    return model.score(test, subset_by_loci=subset_by_loci) 
