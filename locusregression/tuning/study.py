import optuna
import os
from ..corpus import stream_corpus, MetaCorpus, train_test_split
from functools import partial
from ._objective import objective


def _get_nfs_storage(study_name):
    
    journal = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    f'.journal.{study_name}.db'
                )

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal)
    )
    return storage



def create_study(
    locus_subsample = 0.125,
    batch_size = 32,
    factor = 3,
    train_size = 0.7,
    skip_tune_subsample=False,
    model_type = 'regression',
    empirical_bayes = False,
    pi_prior = 1.,
    num_epochs = 500,
    fix_signatures = None,
    use_pruner = False,
    locus_subsample_rates = [0.0625, 0.125, 0.25, 0.5, 1.],
    storage = None,
    seed = 0,*,
    corpuses,
    min_components, 
    max_components,
    study_name,
):
    
    if storage is None:
        storage = _get_nfs_storage(study_name)    

    model_params = dict(
        locus_subsample=locus_subsample,
        batch_size = batch_size,
        pi_prior = pi_prior,
        empirical_bayes = empirical_bayes,
        fix_signatures = fix_signatures,
    )

    if use_pruner:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource = 50,
            reduction_factor = factor,
            max_resource = num_epochs,
            bootstrap_count = 1,
        )
    else:
        pruner = optuna.pruners.NopPruner()


    study = optuna.create_study(
        study_name = study_name,
        storage = storage,
        direction = 'maximize',
        load_if_exists = True,
        pruner = pruner,
        sampler = optuna.samplers.RandomSampler(
            seed = seed,
        ),
    )

    corpuses = [os.path.abspath(corpus) for corpus in corpuses]

    study.set_user_attr('corpuses', corpuses)
    study.set_user_attr('min_components', min_components)
    study.set_user_attr('max_components', max_components)
    study.set_user_attr('model_params', model_params)
    study.set_user_attr('tune_subsample', not skip_tune_subsample)
    study.set_user_attr('locus_subsample_rates', locus_subsample_rates)
    study.set_user_attr('model_type', model_type)
    study.set_user_attr('seed', seed)
    study.set_user_attr('train_size', train_size)
    study.set_user_attr('num_epochs', num_epochs)
    



def load_study(study_name, storage = None, with_corpus = True):
    
    if storage is None:
        storage = _get_nfs_storage(study_name)    

    study = optuna.load_study(
                    study_name=study_name, 
                    storage=storage
                    )

    attrs = study.user_attrs
    corpuses = attrs['corpuses']

    dataset = None
    
    if with_corpus:
        if len(corpuses) == 1:
            dataset = stream_corpus(corpuses[0])
        else:
            dataset = MetaCorpus(*[
                stream_corpus(corpus) for corpus in corpuses
            ])

    return study, dataset, attrs



def run_trial(*,study_name, iters, storage = None):

    study, dataset, attrs = load_study(study_name, storage)

    train, test = train_test_split(
        dataset,
        train_size=attrs['train_size'],
        seed = attrs['seed'],
    )
    
    obj_func = partial(
        objective,
        min_components = attrs['min_components'], 
        max_components = attrs['max_components'],
        model_params = attrs['model_params'],
        tune_subsample = attrs['tune_subsample'],
        locus_subsample_rates = attrs['locus_subsample_rates'],
        model_type = attrs['model_type'],
        num_epochs = attrs['num_epochs'],
        train = train,
        test = test,
    )

    study.optimize(
        obj_func,
        n_trials = iters,
    )

