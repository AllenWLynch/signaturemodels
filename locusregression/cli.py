from ._cli_utils import *
from .corpus import *
from .corpus import logger as reader_logger
from .model import load_model, logger
from .model._importance_sampling import get_posterior_sample
from .tuning import run_trial, create_study, load_study
from .simulation import SimulatedCorpus, coef_l1_distance, signature_cosine_distance
import argparse
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
import pickle
import logging
import warnings
from matplotlib.pyplot import savefig
from .explanation.explanation import explain
from functools import partial
from joblib import Parallel, delayed
import joblib

from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers(help = 'Commands')


def make_windows_wrapper(*,categorical_features, **kw):
    make_windows(*categorical_features, **kw)

make_windows_parser = subparsers.add_parser('get-regions', help = 'Make windows from a genome file.',
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
make_windows_parser.add_argument('--genome-file','-g', type = file_exists, required = True, help = 'Also known as a "Chrom sizes" file.')    
make_windows_parser.add_argument('--blacklist-file','-v', type = file_exists, required=True, help = 'Bed file of regions to exclude from windows.')
make_windows_parser.add_argument('--window-size','-w', type = posint, required = True, help = 'Size of windows to make.')
make_windows_parser.add_argument('--categorical-features','-cf', nargs='+', type = str, default = [], 
                                 help = 'List of categorical feature bedfiles to account for while making windows.')
make_windows_parser.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout, 
                                 help = 'Where to save windows.')
make_windows_parser.set_defaults(func = make_windows_wrapper)


trinuc_sub = subparsers.add_parser('get-trinucs', help = 'Write trinucleotide context file for a given genome.')
trinuc_sub.add_argument('--fasta-file','-fa', type = file_exists, required = True, help = 'Sequence file, used to find context of mutations.')
trinuc_sub.add_argument('--regions-file','-r', type = file_exists, required = True)
trinuc_sub.add_argument('--n-jobs','-j', type = posint, default = 1, help = 'Number of parallel processes to use. Currently does nothing.')
trinuc_sub.add_argument('--output','-o', type = valid_path, required = True, help = 'Where to save compiled corpus.')
trinuc_sub.set_defaults(func = SBSCorpusMaker.create_trinuc_file)


def process_bigwig(group='all',
                   normalization='power',
                   extend=0,*,
                   bigwig_file, 
                   regions_file, 
                   feature_name, 
                   output):

    feature_vals = make_continous_features(
        bigwig_file=bigwig_file,
        regions_file=regions_file,
        extend=extend,
    )

    print('#feature=' + feature_name, file=output)
    print(f'#type={normalization}', file=output)
    print('#group=' + group, file = output)
    print(*feature_vals, sep = '\n', file = output)


bigwig_sub = subparsers.add_parser('ingest-bigwig', help = 'Summarize bigwig file for a given cell type.')
bigwig_sub.add_argument('bigwig-file', type = file_exists)
bigwig_sub.add_argument('--regions-file','-r', type = file_exists, required=True,)
bigwig_sub.add_argument('--feature-name','-name', type = str, required=True,)
bigwig_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
bigwig_sub.add_argument('--extend','-e', type = posint, default=0, help = 'Extend each region by this many basepairs.')
bigwig_sub.add_argument('--normalization','-norm', type = str, choices=['power','minmax','quantile','standardize'], 
                        default='power', 
                        help = 'Normalization to apply to feature.'
                        )
bigwig_sub.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout)
bigwig_sub.set_defaults(func = process_bigwig)


def process_distance_feature(
        group='all',
        normalization='quantile',
        reverse=False,*,
        bed_file,
        regions_file,
        feature_name,
        output,
):
    
    upstream, downstream = make_distance_features(
        genomic_features=bed_file,
        reverse=reverse,
        regions_file=regions_file,
    )

    print(f'#feature={feature_name}_progressBetween\t#feature={feature_name}_interFeatureDistance', file=output)
    print(f'#type=none\t#type={normalization}', file=output)
    print(f'#group={group}\t#group={group}', file = output)
    print(*map(lambda x : '\t'.join(map(str, x)), zip(upstream, downstream)), sep = '\n', file = output)

distance_sub = subparsers.add_parser('ingest-distance', help = 'Summarize distance to nearest feature upstream and downstream for some genomic elements.')
distance_sub.add_argument('bed-file', type = file_exists, help = 'Bed file of genomic features. Only three columns are required, all other columns are ignored.')
distance_sub.add_argument('--regions-file','-r', type = file_exists, required=True,)
distance_sub.add_argument('--feature-name','-name', type = str, required=True,)
distance_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
distance_sub.add_argument('--normalization','-norm', type = str, choices=['power','minmax','quantile','standardize'],
                            default='quantile', help = 'Normalization to apply to feature.')
distance_sub.add_argument('--reverse','-rev', action = 'store_true', default=False, 
                            help = 'Reverse the direction of the distance feature, for instance, if featurizing distance to anti-sense features only.')
distance_sub.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout)
distance_sub.set_defaults(func = process_distance_feature)


def process_discrete(
        group='all',*,
        bed_file,
        regions_file,
        feature_name,
        output,
        null='.',
        class_priority=None,
        column=4,
):

    discrete_features = make_discrete_features(
        genomic_features=bed_file,
        regions_file=regions_file,
        null=null,
        class_priority=class_priority,
        column=column,
    )

    print(f'#feature={feature_name}', file=output)
    print('#type=categorical', file=output)
    print(f'#group={group}', file = output)
    print(*discrete_features, sep = '\n', file = output)


discrete_sub = subparsers.add_parser('ingest-categorical', help = 'Summarize discrete genomic features for some genomic elements.')
discrete_sub.add_argument('bed-file', type = file_exists, help = 'Bed file of genomic features. Only three columns are required, all other columns are ignored.')
discrete_sub.add_argument('--regions-file','-r', type = file_exists, required=True,)
discrete_sub.add_argument('--feature-name','-name', type = str, required=True,)
discrete_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
discrete_sub.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout)
discrete_sub.add_argument('--null','-null', type = str, default='None', help = 'Value to use for missing features.')
discrete_sub.add_argument('--class-priority','-p', type = str, nargs = '+', default=None, help = 'Priority for resolving multiple classes for a single region.')
discrete_sub.add_argument('--column','-c', type = posint, default=4, help = 'Column in bed file to use for feature.')
discrete_sub.set_defaults(func = process_discrete)


def write_dataset(
        weight_col = None,
        chr_prefix = '',
        n_jobs=1,*,
        fasta_file,
        trinuc_file,
        regions_file,
        vcf_files,
        exposure_files,
        correlates_file,
        output,
        corpus_name,
        ):

    shared_args = dict(
        fasta_file = fasta_file, 
        trinuc_file = trinuc_file,
        regions_file = regions_file,
        vcf_files = vcf_files,
        chr_prefix = chr_prefix,
    )

    logging.basicConfig(level=logging.INFO)
    reader_logger.setLevel(logging.INFO)

    if exposure_files is None:
        exposure_files = []

    assert len(exposure_files) in [0,1, len(vcf_files)],\
        'User must provide zero, one, or the number of exposure files which matches the number of VCF files.'

    dataset = CorpusReader.create_corpus(
        **shared_args, 
        weight_col = weight_col,
        exposure_files = exposure_files,
        correlates_file = correlates_file,
        corpus_name = corpus_name,
        n_jobs = n_jobs,
    )

    save_corpus(dataset, output)


dataset_sub = subparsers.add_parser('corpus-make', 
    help= 'Read VCF files and genomic correlates to compile a formatted dataset'
          ' for locus modeling.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

dataset_sub.add_argument('--corpus-name','-n', type = str, required = True, help = 'Name of corpus, must be unique if modeling with other corpuses.')
dataset_sub.add_argument('--vcf-files', '-vcfs', nargs = '+', type = file_exists, required = True,
    help = 'list of VCF files containing SBS mutations.')
dataset_sub.add_argument('--fasta-file','-fa', type = file_exists, required = True, help = 'Sequence file, used to find context of mutations.')
dataset_sub.add_argument('--regions-file','-r', type = file_exists, required = True,
    help = 'Bed file of format with columns (chromosome, start, end) which defines the windows used to represent genomic loci in the model. '
            'The provided regions may be discontinuous, but MUST be in sorted order (run "sort -k1,1 -k2,2 --check <file>").')
dataset_sub.add_argument('--correlates-file', '-c', type = file_exists, required=True,
    help = 'One TSV file, or list of TSV files of genomic correlates. If given as a list, the number of files must match the number of provided VCF files. '
           'The first line must be column names which start with "#". '
           'Each column must have a name, and each TSV file must have the same columns in the same order. '
           'Each row in the file corresponds with the value of those correlates in the analagous region provided in the "regions" file. '
           'Ensure that the correlates are listed in the same order as the regions, and that there are no missing values.'
)
dataset_sub.add_argument('--exposure-files','-e', type = file_exists, nargs = '+',
    help = 'A one-column TSV file or list of. A header is optional. Again, a value must be provided for each region given in the "regions" file, and in the same order. '
           'Exposures are positive scalars which the user calculates to reflect technical influences on the number of mutations one expects to '
           'find within each region. The exposures may be proportional to number of reads falling within each region, or some other '
           'metric to quantify sensitivity to call mutations.')
dataset_sub.add_argument('--trinuc-file','-trinucs', type = file_exists,default=None,
                         help = 'Pre-calculated trinucleotide context file.')
dataset_sub.add_argument('--output','-o', type = valid_path, required = True, help = 'Where to save compiled corpus.')

dataset_sub.add_argument('--weight-col','-w', type = str, default=None,
    help = 'Name of INFO column which contains importance weights per mutation - if not provided all mutations are given a weight of 1. '
           'An example of a useful weight is the tumor cell fraction or relative copy number of that mutation which may be related to local mutation rate due to changes in ploidy. '
           'If the weight column were called INFO/VCN in the VCF, you must only provide --weight-col=VCN.'
)
dataset_sub.add_argument('--n-jobs','-j', type = posint, default = 1,
    help = 'Number of parallel processes to use for reading VCF files.')
dataset_sub.add_argument('--chr-prefix', default= '', help='Append the chromosome names in VCF files with this prefix. Useful if you are using UCSC reference materials.')
dataset_sub.set_defaults(func = write_dataset)
    

def split_corpus(*,corpus, train_output, test_output, train_prop,
                 by_locus = False,
                 seed = 0):

    corpus = load_corpus(corpus)

    train, test = train_test_split(
                    corpus, 
                    train_size=train_prop,
                    seed = seed,
                    by_locus=by_locus
                )

    save_corpus(train, train_output)
    save_corpus(test, test_output)


split_parser = subparsers.add_parser('corpus-split', help='Partition a corpus into training and test sets.')
split_parser.add_argument('corpus', type = file_exists)
split_parser.add_argument('--train-output','-to', type = valid_path, required=True)
split_parser.add_argument('--test-output', '-vo', type = valid_path, required=True)
split_parser.add_argument('--train-prop', '-p', type = posfloat, default=0.7)
split_parser.add_argument('--by-locus', action = 'store_true', default=False,
                            help = 'Split by locus instead of by sample.')
split_parser.add_argument('--seed', '-s', type = posint, default=0)
split_parser.set_defaults(func = split_corpus)


def empirical_mutation_rate(*,corpus, output):
    mutation_rate = load_corpus(corpus).get_empirical_mutation_rate()
    print(*mutation_rate, file = output, sep = '\n')

empirical_mutrate_parser = subparsers.add_parser('corpus-empirical-mutation-rate',
    help = 'Aggregate mutations in a corpus to calculate the log (natural) empirical mutation rate. This depends on having sufficient mutations to find a smooth function.'
)
empirical_mutrate_parser.add_argument('corpus', type = file_exists)
empirical_mutrate_parser.add_argument('--output', '-o', type = argparse.FileType('w'), 
                                        default = sys.stdout)
empirical_mutrate_parser.set_defaults(func = empirical_mutation_rate)


def _overwrite_features_helper(*, corpus, correlates_file):
    features, feature_names = CorpusReader.read_correlates(correlates_file)
    overwrite_corpus_features(corpus, features.T, feature_names)

overwrite_features_parser = subparsers.add_parser('corpus-overwrite-features',
    help = 'Overwrite the feature matrix of a corpus with a new one.'
)
overwrite_features_parser.add_argument('corpus', type = file_exists)
overwrite_features_parser.add_argument('--correlates-file','-c', type = file_exists, required=True,
                                       help = 'TSV file of genomic correlates. The first line must be column names which start with "#".')
overwrite_features_parser.set_defaults(func = _overwrite_features_helper)


tune_sub = subparsers.add_parser('study-create', 
    help = 'Tune number of signatures for LocusRegression model on a pre-compiled corpus using the'
    'hyperband or successive halving algorithm.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

tune_required = tune_sub.add_argument_group('Required arguments')
tune_required.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
tune_required.add_argument('--study-name','-sn', type = str, required=True,
                           help = 'Name of study, used to store tuning results in a database.')
tune_required.add_argument('--max-components','-max',type = posint, required=True,
    help= 'Maximum number of components to test for fit on dataset.')
tune_required.add_argument('--min-components','-min',type = posint, required=True,
    help= 'Maximum number of components to test for fit on dataset.')

tune_optional = tune_sub.add_argument_group('Tuning arguments')

tune_optional.add_argument('--storage',type = str, default=None,
                            help = 'Address path to database to store tuning results, for example "sqlite:///tuning.db" '
                                   'or "mysql://user:password@host:port/dbname", if one is using a remote database. '
                                   'If left unset, will store tuning results in a local file.'
                            )
tune_optional.add_argument('--factor','-f',type = posint, default = 4,
    help = 'Successive halving reduction factor for each iteration')
tune_optional.add_argument('--skip-tune-subsample', action = 'store_true', default=False)
#tune_optional.add_argument('--locus-subsample-rates','-rates', type = posfloat, nargs = '+', default = [0.0625, 0.125, 0.25, 0.5, 1])
tune_optional.add_argument('--use-pruner', '-prune', action='store_true', default=False,
                           help = 'Use the hyperband pruner to eliminate poorly performing trials before they complete, saving computational resources.')

model_options = tune_sub.add_argument_group('Model arguments')

model_options.add_argument('--fix-signatures','-sigs', nargs='+', type = str, default = None,
                              help = 'COSMIC signatures to fix as part of the generative model, referenced by name (SBS1, SBS2, etc.)\n '
                                    'The number of signatures listed must be less than or equal to the number of components.\n '
                                    'Any extra components will be initialized randomly and learned. If no signatures are provided, '
                                    'all are learned de-novo.'
                              )
model_options.add_argument('--num-epochs', '-e', type = posint, default=500,
    help = 'Maximum number of epochs to allow training during'
            ' successive halving/Hyperband. This should be set high enough such that the model converges to a solution.')
model_options.add_argument('--model-type','-model', choices=['linear','gbt'], default='linear')
model_options.add_argument('--locus-subsample','-sub', type = posfloat, default = 0.125,
    help = 'Whether to use locus subsampling to speed up training via stochastic variational inference.')
model_options.add_argument('--batch-size','-batch', type = posint, default = 128,
    help = 'Batch size for stochastic variational inference.')
model_options.add_argument('--empirical-bayes','-eb', action = 'store_true', default=False,)
model_options.add_argument('--pi-prior', '-pi', type = posfloat, default = 1.,
    help = 'Dirichlet prior over sample mixture compositions. A value > 1 will give more dense compositions, which <1 finds more sparse compositions.')
tune_sub.set_defaults(func = create_study)


def wraps_run_trial(**kw):
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    run_trial(**kw)

trial_parser = subparsers.add_parser('study-run-trial', help='Run a single trial of hyperparameter tuning.')
trial_parser.add_argument('study-name',type = str)
trial_parser.add_argument('--storage','-s', type = str, default=None)
trial_parser.add_argument('--iters','-i', type = posint, default=1)
trial_parser.set_defaults(func = wraps_run_trial)


def summarize_study(*,study_name, output, storage = None):

    study, *_ = load_study(study_name, storage)
    
    study.trials_dataframe().to_csv(output, index = False)


summarize_parser = subparsers.add_parser('study-summarize', help = 'Summarize tuning results from a study.')
summarize_parser.add_argument('study-name',type = str)
summarize_parser.add_argument('--output','-o', type = valid_path, required=True)
summarize_parser.add_argument('--storage','-s', type = str, default=None)
summarize_parser.set_defaults(func = summarize_study)


def retrain_best(trial_num = None,
                 storage = None, 
                 verbose = False,
                 num_epochs = None,*,
                 study_name, output):
    
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    study, dataset, attrs = load_study(study_name, storage)

    if trial_num is None:
        best_trial = study.best_trial
    else:
        best_trial = study.trials[trial_num]

    model_params = attrs['model_params']
    model_params.update(best_trial.params)

    basemodel = _get_basemodel(attrs["model_type"])
    
    print(
        'Training model with params:\n' + \
        '\n'.join(
            [f'\t{k} = {v}' for k,v in model_params.items()]
        ),
        file = sys.stderr,
    )

    model = basemodel(
            eval_every = 1000000,
            quiet= not verbose,
            **model_params,
            seed = best_trial.number,
            num_epochs= num_epochs or attrs['num_epochs'],
        )

    model.fit(dataset)

    model.save(output)


retrain_sub = subparsers.add_parser('study-retrain', help = 'From tuning results, retrain a chosen or the best model.')
retrain_sub.add_argument('study-name',type = str)
retrain_sub.add_argument('--storage','-s', type = str, default=None)
retrain_sub.add_argument('--verbose','-v',action = 'store_true', default = False,)
retrain_sub .add_argument('--output','-o', type = valid_path, required=True,
    help = 'Where to save trained model.')
retrain_sub.add_argument('--trial-num','-t', type = posint, default=None,
    help= 'If left unset, will retrain model with best params from tuning results.\nIf provided, will retrain model parameters from the "trial_num"th trial.')
retrain_sub.add_argument('--num-epochs','-epochs', type = int, default = None, 
    help='Override the number of epochs used for tuning.')

retrain_sub.set_defaults(func = retrain_best)

def train_model(
        locus_subsample = 0.125,
        batch_size = 128,
        time_limit = None,
        tau = 16,
        kappa = 0.5,
        seed = 0, 
        pi_prior = 1.,
        num_epochs = 10000, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        eval_every = 20,
        bound_tol = 1e-2,
        verbose = False,
        n_jobs = 1,
        empirical_bayes = True,
        model_type = 'linear',
        begin_prior_updates = 10,
        fix_signatures = None,*,
        n_components,
        corpuses,
        output,
    ):

    basemodel = _get_basemodel(model_type)
    
    model = basemodel(
        fix_signatures=fix_signatures,
        locus_subsample = locus_subsample,
        batch_size = batch_size,
        seed = seed, 
        pi_prior = pi_prior,
        num_epochs = num_epochs, 
        difference_tol = difference_tol,
        estep_iterations = estep_iterations,
        quiet = not verbose,
        bound_tol = bound_tol,
        n_components = n_components,
        n_jobs= n_jobs,
        time_limit=time_limit,
        eval_every = eval_every,
        tau = tau,
        kappa = kappa,
        empirical_bayes=empirical_bayes,
        begin_prior_updates=begin_prior_updates,
    )
    
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    dataset = _load_dataset(corpuses)
    
    model.fit(dataset)
    
    model.save(output)



trainer_sub = subparsers.add_parser('model-train', 
                                    help = 'Train LocusRegression model on a pre-compiled corpus.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

trainer_required = trainer_sub.add_argument_group('Required arguments')
trainer_required .add_argument('--n-components','-k', type = posint, required=True,
    help = 'Number of signatures to learn.')
trainer_required .add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
trainer_required .add_argument('--output','-o', type = valid_path, required=True,
    help = 'Where to save trained model.')

trainer_optional = trainer_sub.add_argument_group('Optional arguments')

trainer_optional.add_argument('--model-type','-model', choices=['linear','gbt'], default='linear')
trainer_optional.add_argument('--locus-subsample','-sub', type = posfloat, default = None,
    help = 'Whether to use locus subsampling to speed up training via stochastic variational inference.')
trainer_optional.add_argument('--batch-size','-batch', type = posint, default = 100000,
    help = 'Use minibatch updates via stochastic variational inference.')
trainer_optional.add_argument('--begin-prior-updates', type = int, default=10)
trainer_optional.add_argument('--time-limit','-time', type = posint, default = None,
    help = 'Time limit in seconds for model training.')
trainer_optional.add_argument('--fix-signatures','-sigs', nargs='+', type = str, default = None,
                              help = 'COSMIC signatures to fix as part of the generative model, referenced by name (SBS1, SBS2, etc.)\n '
                                    'The number of signatures listed must be less than or equal to the number of components.\n '
                                    'Any extra components will be initialized randomly and learned. If no signatures are provided, '
                                    'all are learned de-novo.'
                              )
trainer_optional.add_argument('--empirical-bayes','-eb', action = 'store_true', default=False,)
trainer_optional.add_argument('--tau', type = posint, default = 16)
trainer_optional.add_argument('--kappa', type = posfloat, default=0.5)
trainer_optional.add_argument('--eval-every', '-eval', type = posint, default = 10,
    help = 'Evaluate the bound after every this many epochs')
trainer_optional.add_argument('--seed', type = posint, default=1776)
trainer_optional.add_argument('--pi-prior','-pi', type = posfloat, default = 1.,
    help = 'Dirichlet prior over sample mixture compositions. A value > 1 will give more dense compositions, which <1 finds more sparse compositions.')
trainer_optional.add_argument('--num-epochs', '-epochs', type = posint, default = 1000,
    help = 'Maximum number of epochs to train.')
trainer_optional.add_argument('--bound-tol', '-tol', type = posfloat, default=1e-2,
    help = 'Early stop criterion, stop training if objective score does not increase by this much after one epoch.')
trainer_optional.add_argument('--verbose','-v',action = 'store_true', default = False,)
trainer_sub.set_defaults(func = train_model)



def score(*,model, corpuses):

    dataset = _load_dataset(corpuses)

    model = load_model(model)

    print(model.score(dataset))


score_parser = subparsers.add_parser('model-score', help='Score a model on a corpus.')
score_parser.add_argument('model', type = file_exists)
score_parser.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
score_parser.set_defaults(func = score)



def predict(*,model, corpuses, output):

    dataset = _load_dataset(corpuses)

    model = load_model(model)

    exposures_matrix = model.predict(dataset)

    exposures_matrix.to_csv(output)

predict_sub = subparsers.add_parser('model-predict', help = 'Predict exposures for each sample in a corpus.')
predict_sub.add_argument('model', type = file_exists)
predict_sub.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
                         help = 'Path to compiled corpus file/files.')
predict_sub.add_argument('--output','-o', type =  valid_path, required=True)
predict_sub.set_defaults(func = predict)


def summary_plot(*,model, output):
    model = load_model(model)
    model.plot_summary()
    savefig(output, bbox_inches='tight', dpi = 300)

summary_plot_parser = subparsers.add_parser('model-plot-summary', help = 'Plot summary of model components.')
summary_plot_parser.add_argument('model', type = file_exists)
summary_plot_parser.add_argument('--output','-o', type = valid_path, required=True,
                                 help = 'Path to save plot, the file extension determines the format')
summary_plot_parser.set_defaults(func = summary_plot)


def save_signatures(*, model, output):
    
    model = load_model(model)
    
    print('', *COSMIC_SORT_ORDER, sep = ',', file = output)
    for i, component_name in enumerate(model.component_names):
        print(component_name, *model.signature(i, return_error=False), sep = ',', file = output)

signatures_parser = subparsers.add_parser('model-save-signatures', 
                                          help = 'Save signatures to file.')
signatures_parser.add_argument('model', type = file_exists)
signatures_parser.add_argument('--output','-o', type =  argparse.FileType('w'), default=sys.stdout)
signatures_parser.set_defaults(func = save_signatures)


def list_corpuses(*,model):
    model = load_model(model)
    print(*model.corpus_states.keys(), sep = '\n', file = sys.stdout)

list_corpuses_parser = subparsers.add_parser('model-list-corpuses',
                                             help = 'List the names of corpuses used to train a model.')
list_corpuses_parser.add_argument('model', type = file_exists)
list_corpuses_parser.set_defaults(func = list_corpuses)


def get_mutation_rate_r2(*, model, corpuses):

    model = load_model(model)
    dataset = _load_dataset(corpuses)
    
    print(
        model.get_mutation_rate_r2(dataset),
        file = sys.stdout
    )

mutrate_r2_parser = subparsers.add_parser('model-mutation-rate-r2',
                                            help = 'Calculate the R^2 of the model\'s predicted mutation rate w.r.t. the empirical mutation rate.')
mutrate_r2_parser.add_argument('model', type = file_exists)
mutrate_r2_parser.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
mutrate_r2_parser.set_defaults(func = get_mutation_rate_r2)


def explain_wrapper(n_jobs=1,*,signature, model, corpuses, output):
    
    model = load_model(model)
    dataset = _load_dataset(corpuses)

    results = explain(signature,
            model = model,
            corpus = dataset,
            n_jobs = n_jobs
            )
    
    print(*results['feature_names'], sep=',', file = output)
    for row in results['shap_values']:
        print(*row, sep = ',', file = output)

explain_parser = subparsers.add_parser('model-explain',
                                        help = 'Explain the contribution of each feature to a signature.')
explain_parser.add_argument('model', type = file_exists)
explain_parser.add_argument('--signature','-sig', type = str, required=True)
explain_parser.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
explain_parser.add_argument('--n-jobs','-j', type = posint, default = 1)
explain_parser.add_argument('--output','-o', type =  argparse.FileType('w'), default=sys.stdout)
explain_parser.set_defaults(func = explain_wrapper)



def _make_corpusstate_cache(*,model, corpus):

    cache_path = get_corpusstate_cache_path(model, corpus)

    model = load_model(model)
    corpus = stream_corpus(corpus)

    corpus_state = model._init_new_corpusstates(corpus)[corpus.name]

    joblib.dump(corpus_state, cache_path)

corpusstate_cache_parser = subparsers.add_parser('model-cache-sstats')
corpusstate_cache_parser.add_argument('model', type = file_exists)
corpusstate_cache_parser.add_argument('--corpus','-d', type = file_exists, required=True)
corpusstate_cache_parser.set_defaults(func = _make_corpusstate_cache)



def _write_posterior_annotated_vcf(
    input_vcf, output,*,
    model_state, 
    weight_col, 
    corpus_state,
    regions_file,
    fasta_file,
    chr_prefix,
    component_names,
    ):

    sample = SBSCorpusMaker.ingest_sample(
                    input_vcf, 
                    regions_file=regions_file,
                    fasta_file=fasta_file,
                    chr_prefix=chr_prefix,
                    weight_col=weight_col,
                )
    
    posterior_df = get_posterior_sample(
                        sample=sample,
                        model_state=model_state,
                        corpus_state=corpus_state,
                        component_names=component_names,
                        n_iters=5000,
                    )

    transfer_annotations_to_vcf(
        posterior_df,
        vcf_file=input_vcf,
        description='Log-probability under the posterior that the mutation was generated by this process.', 
        output=output, 
        chr_prefix=chr_prefix,
    )


def assign_components_wrapper(*,
        model, 
        vcf_files, 
        corpus, 
        output_prefix, 
        exposure_file=None, 
        weight_col=None,
        chr_prefix = '',
        n_jobs=1,
    ):

    corpus_state = load_corpusstate_cache(model, corpus)
    
    model = load_model(model)
    corpus = stream_corpus(corpus)

    try:
        corpus.metadata['regions_file']
    except KeyError as err:
        raise ValueError('Corpus must have a "regions_file" metadata attribute.\n'
                         'This one doesn\'t, which means it must be a partition of some corpus, or a simluted corpus.\n'
                         'This function must use the originally-created corpus.') from err

    annotation_fn = partial(
        _write_posterior_annotated_vcf,
        model_state=model.model_state, 
        weight_col=weight_col, 
        corpus_state=corpus_state,
        regions_file=corpus.metadata['regions_file'],
        fasta_file=corpus.metadata['fasta_file'],
        chr_prefix=chr_prefix,
        component_names=model.component_names,
    )

    del corpus

    Parallel(
            n_jobs = n_jobs, 
            verbose = 10,
        )(
            delayed(annotation_fn)(vcf, output_prefix + os.path.basename(vcf))
            for vcf in vcf_files
        )

    
assign_components_parser = subparsers.add_parser('model-annotate-mutations',
    help = 'Assign each mutation in a VCF file to a component of the model.')
assign_components_parser.add_argument('model', type = file_exists)
assign_components_parser.add_argument('--vcf-files','-vcfs', nargs='+', type = file_exists, required=True)
assign_components_parser.add_argument('--corpus','-d', type = file_exists, required=True)
assign_components_parser.add_argument('--output-prefix','-prefix', type = str, required=True)
assign_components_parser.add_argument('--exposure-file','-e', type = file_exists, default=None)
assign_components_parser.add_argument('--chr-prefix', type = str, default = '')
assign_components_parser.add_argument('--weight-col','-w', type = str, default=None)
assign_components_parser.add_argument('--n-jobs','-j',type=posint, default=1)
assign_components_parser.set_defaults(func = assign_components_wrapper)


'''
def summarize_all_model_attributes(*,model, prefix):

    corpuses = list(load_model(model).corpus_states.keys())
    with open(prefix + '.associations.csv', 'w') as associations_file:
        save_associations(model = model, output = associations_file)

    with open(prefix + '.signatures.csv', 'w') as signatures_file:
        save_signatures(model = model, output = signatures_file)
    
    summary_plot(model = model, output = prefix + '.summary_plt.png')

    for corpus in corpuses:
        with open(prefix + f'.mutrates.{corpus}.csv', 'w') as mutrates_file:
            save_per_component_mutation_rates(model = model, output = mutrates_file, corpus_name = corpus)
        
        with open(prefix + f'.overall_mutrate.{corpus}.csv', 'w') as mutrate_file:
            save_overall_mutation_rate(model = model, output = mutrate_file, corpus_name = corpus)



summarize_parser = subparsers.add_parser('model-summarize', help = 'Save all summary data for a trained model.')
summarize_parser.add_argument('model', type = file_exists)
summarize_parser.add_argument('--prefix','-p', type = str, required=True)
summarize_parser.set_defaults(func = summarize_all_model_attributes)
'''

'''
def add_sample_ingest_args(parser):
    parser.add_argument('model', type = file_exists)
    parser.add_argument('--vcf-file','-vcf', type = file_exists, required=True)
    parser.add_argument('--exposure-file','-e', type = argparse.FileType('r'), default=None)
    parser.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--regions-file','-r', type = file_exists, required=True)
    parser.add_argument('--fasta-file','-fa', type = file_exists, required=True)
    parser.add_argument('--chr-prefix', type = str, default = '')


def assign_components(*,model,vcf_file, corpus_name, exposure_file, output,
                     regions_file, fasta_file, 
                     chr_prefix = ''):
    
    model = load_model(model)

    try:
        model.corpus_states[corpus_name]
    except KeyError:
        raise ValueError(f'Corpus {corpus_name} not found in model.')
    
    logger.info('Reading sample from VCF file.')
    sample = CorpusReader.ingest_sample(
        vcf_file, exposure_file = exposure_file,
        regions_file = regions_file, fasta_file = fasta_file,
        chr_prefix = chr_prefix,
    )
    
    mutation_assignments = model.assign_mutations_to_components(sample, corpus_name)

    print(*mutation_assignments.keys(), sep = ',', file = output)
    for row in list(zip(*mutation_assignments.values())):
        print(*row, sep = ',', file = output)


assign_mutations_parser = subparsers.add_parser('model-assign-components',
                                                help = 'Assign each mutation in a VCF file to a component of the model.')
add_sample_ingest_args(assign_mutations_parser)
assign_mutations_parser.add_argument('--corpus-name','-n', type = str, required=True)
assign_mutations_parser.set_defaults(func = assign_components)


def assign_corpus(*,model,vcf_file, exposure_file, output,
                     regions_file, fasta_file, 
                     iters = 100, anneal_steps = 100,
                     max_mutations = 10000,
                     chr_prefix = '',
                     pi_prior = None):
    
    model = load_model(model)
    
    logger.info('Reading sample from VCF file.')
    sample = CorpusReader.ingest_sample(
        vcf_file, exposure_file = exposure_file,
        regions_file = regions_file, fasta_file = fasta_file,
        chr_prefix = chr_prefix,
    )
    
    corpus_logps = model.assign_sample_to_corpus(sample, 
                                                 n_iters = iters, 
                                                 n_samples_per_iter = anneal_steps,
                                                 pi_prior = pi_prior,
                                                 max_mutations = max_mutations,
                                                )

    print(*corpus_logps.keys(), sep = ',', file = output)
    for row in list(zip(*corpus_logps.values())):
        print(*row, sep = ',', file = output)


assign_corpus_parser = subparsers.add_parser('model-assign-corpus-sample',
                                             help = 'Evaluate which corpus a VCF file is most likely generated from.')
add_sample_ingest_args(assign_corpus_parser)
assign_corpus_parser.add_argument('--pi-prior','-pi', type = posfloat, default = None)
assign_corpus_parser.add_argument('--iters','-iters', type = posint, default = 100)
assign_corpus_parser.add_argument('--anneal-steps','-steps', type = posint, default = 100)
assign_corpus_parser.add_argument('--max-mutations','-max', type = posint, default = 10000)
assign_corpus_parser.set_defaults(func = assign_corpus)


def assign_corpus_mutation(*,model,vcf_file, exposure_file, output,
                        regions_file, fasta_file, iters = 100, anneal_steps = 100,
                        chr_prefix = ''):
    
    model = load_model(model)
    
    logger.info('Reading sample from VCF file.')
    sample = CorpusReader.ingest_sample(
        vcf_file, exposure_file = exposure_file,
        regions_file = regions_file, fasta_file = fasta_file,
        chr_prefix = chr_prefix,
    )

    mutation_assignments = model.assign_mutations_to_corpus(sample, 
                                                 n_iters = iters, 
                                                 n_samples_per_iter = anneal_steps
                                                )
    
    print(*mutation_assignments.keys(), sep = ',', file = output)
    for row in list(zip(*mutation_assignments.values())):
        print(*row, sep = ',', file = output)
    
assign_corpus_mutation_parser = subparsers.add_parser('model-assign-corpus-mutation',
                                                        help = 'Evaluate which corpus a mutation in a VCF file is most likely generated from.')
add_sample_ingest_args(assign_corpus_mutation_parser)
assign_corpus_mutation_parser.add_argument('--iters','-iters', type = posint, default = 100)
assign_corpus_mutation_parser.add_argument('--anneal-steps','-steps', type = posint, default = 100)
assign_corpus_mutation_parser.set_defaults(func = assign_corpus_mutation)
'''

def run_simulation(*,config, prefix):
    
    with open(config, 'rb') as f:
        configuration = pickle.load(f)

    corpus_path = prefix + 'corpus.h5'
    params_path = prefix + 'generative_params.pkl'
    assert valid_path(corpus_path)
    assert valid_path(params_path)

    corpus, generative_parameters = SimulatedCorpus.create(
        **configuration
    )

    save_corpus(corpus, corpus_path)

    with open(params_path, 'wb') as f:
        pickle.dump(generative_parameters, f)
    


def simulate_from_model(*, model, corpus, output, 
                        seed=0, use_signatures=None,
                        n_jobs=1,):
    
    model = load_model(model)
    corpus = stream_corpus(corpus)

    corpus_state = model._init_new_corpusstates(corpus)[corpus.name]

    resampled_corpus = SimulatedCorpus.from_model(
        model = model,
        corpus_state = corpus_state,
        corpus = corpus,
        use_signatures = use_signatures,
        seed = seed,
        n_jobs = n_jobs,
    )

    save_corpus(resampled_corpus, output)

simulate_from_model_parser = subparsers.add_parser('simulate-from-model',
                                                    help = 'Simulate a new corpus from a trained model.')
simulate_from_model_parser.add_argument('model', type = file_exists)
simulate_from_model_parser.add_argument('--corpus','-d', type = file_exists, required=True)
simulate_from_model_parser.add_argument('--output','-o', type = valid_path, required=True)
simulate_from_model_parser.add_argument('--use-signatures','-sigs', nargs='+', type = str, default=None)
simulate_from_model_parser.add_argument('--seed', type = posint, default=0)
simulate_from_model_parser.add_argument('--n-jobs', '-j', type = posint, default=1)
simulate_from_model_parser.set_defaults(func = simulate_from_model)


simulation_sub = subparsers.add_parser('simulate', help = 'Create a simulated dataset of cells with mutations')
simulation_sub.add_argument('--config','-c', type = file_exists, required=True,
        help = 'Pickled configuration file for the simulation.'
)
simulation_sub.add_argument('--prefix','-p', type = str, required=True,
        help = 'Prefix under which to save generative params and corpus.'
)
simulation_sub.set_defaults(func = run_simulation)


def evaluate_model(*,simulation_params, model):

    model = load_model(model)

    with open(simulation_params, 'rb') as f:
        params = pickle.load(f)

    coef_l1 = coef_l1_distance(model, params)
    signature_cos = signature_cosine_distance(model, params)

    print('coef_L1_dist','signature_cos_sim', sep = '\t')
    print(coef_l1, signature_cos, sep = '\t')


eval_sub = subparsers.add_parser('eval-sim', help = 'Evaluate a trained model against a simulation\'s generative parameters.')
eval_sub.add_argument('--simulation-params','-sim', type = file_exists, required=True,
        help = 'File path to generative parameters of simulation')

eval_sub.add_argument('--model','-m', type = file_exists, required=True,
        help = 'File path to model.')

eval_sub.set_defaults(func = evaluate_model)


def main():
    #____ Execute commands ___

    logger.setLevel(logging.INFO)

    args = parser.parse_args()

    try:
        args.func #first try accessing the .func attribute, which is empty if user tries ">>>lisa". In this case, don't throw error, display help!
    except AttributeError:
        parser.print_help(sys.stderr)
        
    else:
        
        args = vars(args)
        func = args.pop('func')
        args = {k.replace('-','_') : v for k,v in args.items()}

        func(**args)
