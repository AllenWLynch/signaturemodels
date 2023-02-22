from .corpus import CorpusReader, save_corpus, stream_corpus, MetaCorpus
from .model import LocusRegressor, tune_model, load_model
from .simulation import SimulatedCorpus, coef_l1_distance, signature_cosine_distance
import argparse
from argparse import ArgumentTypeError
import os
import numpy as np
import sys
import logging
import json
import pickle

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def posint(x):
    x = int(x)

    if x > 0:
        return x
    else:
        raise ArgumentTypeError('Must be positive integer.')


def posfloat(x):
    x = float(x)
    if x > 0:
        return x
    else:
        raise ArgumentTypeError('Must be positive float.')


def file_exists(x):
    if os.path.exists(x):
        return x
    else:
        raise ArgumentTypeError('File {} does not exist.'.format(x))

def valid_path(x):
    if not os.access(x, os.W_OK):
        
        try:
            open(x, 'w').close()
            os.unlink(x)
            return x
        except OSError as err:
            raise ArgumentTypeError('File {} cannot be written. Invalid path.'.format(x)) from err
    
    return x


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers(help = 'commands')


def write_dataset(
        sep = '\t', 
        index = -1,*,
        fasta_file,
        genome_file,
        regions_file,
        vcf_files,
        exposure_files,
        correlates_file,
        output
        ):

    shared_args = dict(
        fasta_file = fasta_file, 
        genome_file = genome_file,
        regions_file = regions_file,
        vcf_files = vcf_files,
        sep = sep, index = index,
    )

    if exposure_files is None:
        exposure_files = []

    assert len(exposure_files) in [0,1, len(vcf_files)],\
        'User must provide zero, one, or the number of exposure files which matches the number of VCF files.'

    dataset = CorpusReader.create_corpus(
        **shared_args, 
        exposure_files = exposure_files,
        correlates_file = correlates_file,
    )

    save_corpus(dataset, output)


dataset_sub = subparsers.add_parser('make-corpus', 
    help= 'Read VCF files and genomic correlates to compile a formatted dataset'
          ' for locus modeling.'
    )
dataset_sub.add_argument('--vcf-files', '-vcf', nargs = '+', type = file_exists, required = True,
    help = 'list of VCF files containing SBS mutations.')
dataset_sub.add_argument('--fasta-file','-fa', type = file_exists, required = True, help = 'Sequence file, used to find context of mutations.')
dataset_sub.add_argument('--genome-file','-g', type = file_exists, required = True, 
    help = 'Also known as a "chromsizes" file. Gives the name and length of each chromosome.')
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

dataset_sub.add_argument('--output','-o', type = valid_path, required = True, help = 'Where to save compiled corpus.')

dataset_sub.add_argument('--sep','-sep',default ='\t',type = str, help = 'Separator for VCF file.')
dataset_sub.add_argument('--index','-i',default = -1,type = int, 
                         help = 'Position offset between VCF and Fasta/Bed files. If VCF files were written with 1-based indexing,'
                                'set this to "-1".')

dataset_sub.set_defaults(func = write_dataset)
    
    
def train_model(
        locus_subsample = 0.125,
        time_limit = None,
        tau = 1,
        kappa = 0.5,
        seed = 0, 
        pi_prior = 5.,
        num_epochs = 10000, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        eval_every = 10,
        bound_tol = 1e-2,
        quiet = True,
        n_jobs = 1,*,
        n_components,
        corpuses,
        output,
    ):
    
    model = LocusRegressor(
        locus_subsample = locus_subsample,
        seed = seed, 
        dtype = np.float32,
        pi_prior = pi_prior,
        num_epochs = num_epochs, 
        difference_tol = difference_tol,
        estep_iterations = estep_iterations,
        bound_tol = bound_tol,
        quiet = quiet,
        n_components = n_components,
        n_jobs= n_jobs,
        time_limit=time_limit,
        eval_every = eval_every,
        tau = tau,
        kappa = kappa,
    )
    
    if len(corpuses) == 1:
        dataset = stream_corpus(corpuses[0])
    else:
        dataset = MetaCorpus(*[
            stream_corpus(corpus) for corpus in corpuses
        ])

    logging.basicConfig( level=logging.INFO )
    
    model.fit(dataset)
    
    model.save(output)



trainer_sub = subparsers.add_parser('train-model', help = 'Train LocusRegression model on a pre-compiled corpus.')

trainer_required = trainer_sub.add_argument_group('Required arguments')
trainer_required .add_argument('--n-components','-k', type = posint, required=True,
    help = 'Number of signatures to learn.')
trainer_required .add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
trainer_required .add_argument('--output','-o', type = valid_path, required=True,
    help = 'Where to save trained model.')

trainer_optional = trainer_sub.add_argument_group('Optional arguments')

trainer_optional.add_argument('--locus-subsample','-sub', type = posfloat, default = 0.125,
    help = 'Whether to use locus subsampling to speed up training via stochastic variational inference.')
trainer_optional.add_argument('--time-limit','-time', type = posint, default = None,
    help = 'Time limit in seconds for model training.')
trainer_optional.add_argument('--tau', type = posint, default = 1)
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
trainer_sub.set_defaults(func = train_model)
    

def tune(
    locus_subsample = 0.125,
    batch_size = 128,
    n_jobs = 1,
    factor = 3,
    train_size = 0.7,
    max_time = None,
    tune_subsample= False,*,
    output,
    corpuses,
    min_components, 
    max_components,
):

    model_params = dict(
        locus_subsample=locus_subsample,
        batch_size = batch_size,
    )
    
    if len(corpuses) == 1:
        dataset = stream_corpus(corpuses[0])
    else:
        dataset = MetaCorpus(*[
            stream_corpus(corpus) for corpus in corpuses
        ])

    results = tune_model(
        dataset,
        n_jobs = n_jobs,
        train_size = train_size,
        min_components = min_components,
        max_components = max_components,
        factor = factor,
        max_time=max_time,
        tune_subsample= tune_subsample,
        **model_params,
    )

    with open(output,'w') as f:
        json.dump(results, f)
    

tune_sub = subparsers.add_parser('tune', help = 'Tune number of signatures for LocusRegression model on a pre-compiled corpus using the'
    'Successive Halving algorithm.')

tune_required = tune_sub.add_argument_group('Required arguments')
tune_required.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
tune_required.add_argument('--output','-o', type = valid_path, required=True,
    help = 'Where to save tuning results.')
tune_required.add_argument('--max-components','-max',type = posint, required=True,
    help= 'Maximum number of components to test for fit on dataset.')
tune_required.add_argument('--min-components','-min',type = posint, required=True,
    help= 'Maximum number of components to test for fit on dataset.')
tune_required.add_argument('--n-jobs','-j', type = posint, required= True,
    help = 'Number of concurrent jobs to run.')

tune_optional = tune_sub.add_argument_group('Optional arguments')

tune_sub.add_argument('--max-time', '-t', type = posint, default=None,
    help = 'Maximum length of time to allow training during'
            ' successive halving/Hyperband. This should be set high enough such that the model converges to a solution.')
tune_optional.add_argument('--factor','-f',type = posint, default = 3,
    help = 'Successive halving reduction factor for each iteration')
tune_optional.add_argument('--tune-subsample', action = 'store_true', default=False)

model_options = tune_sub.add_argument_group('Model arguments')

model_options.add_argument('--locus-subsample','-sub', type = posfloat, default = 0.125,
    help = 'Whether to use locus subsampling to speed up training via stochastic variational inference.')
model_options.add_argument('--batch-size','-batch', type = posfloat, default = 128,
    help = 'Batch size for stochastic variational inference.')

tune_sub.set_defaults(func = tune)



def retrain_best(trial_num = None,*,
    tune_results, corpuses, output):

    if len(corpuses) == 1:
        dataset = stream_corpus(corpuses[0])
    else:
        dataset = MetaCorpus(*[
            stream_corpus(corpus) for corpus in corpuses
        ])

    logging.basicConfig( level=logging.INFO )

    with open(tune_results, 'r') as f:
        results = json.load(f)

    def extract_params(trial):
        return {param[6:] : v for param, v in trial.items() if param[:6] == 'param_'}

    max_resoures = max([r['resources'] for r in results])

    if trial_num is None:
        fully_trained = [r for r in results if r['resources'] == max_resoures and r['score'] < 0]
        best_trial = sorted(fully_trained, key = lambda r : r['score'])[-1]
    else:
        best_trial = results[trial_num]

    params = extract_params(best_trial)
    param_string = '\n\t'.join([f'{param}: {v}' for param, v in params.items()])

    print('Training model with params:\n\t' + param_string)

    model = LocusRegressor(**params, time_limit=max_resoures).fit(dataset)

    model.save(output)

retrain_sub = subparsers.add_parser('retrain', help = 'From tuning results, retrain a chosen or the best model.')
retrain_sub .add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
retrain_sub .add_argument('--output','-o', type = valid_path, required=True,
    help = 'Where to save trained model.')
retrain_sub.add_argument('--tune-results','-r',type = file_exists, required=True,
    help = 'Filepath to tuning results.')
retrain_sub.add_argument('--trial-num','-t', type = posint, default=None,
    help= 'If left unset, will retrain model with best params from tuning results.\nIf provided, will retrain model parameters from the "trial_num"th trial.')

retrain_sub.set_defaults(func = retrain_best)


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

    print('coef_L1_dist','signature_cos_dist', sep = '\t')
    print(coef_l1, signature_cos, sep = '\t')


eval_sub = subparsers.add_parser('eval-sim', help = 'Evaluate a trained model against a simulation\'s generative parameters.')
eval_sub.add_argument('--simulation-params','-sim', type = file_exists, required=True,
        help = 'File path to generative parameters of simulation')

eval_sub.add_argument('--model','-m', type = file_exists, required=True,
        help = 'File path to model.')

eval_sub.set_defaults(func = evaluate_model)


def main():
    #____ Execute commands ___

    logging.basicConfig(level = logging.INFO)

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