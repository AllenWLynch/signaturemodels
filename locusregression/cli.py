from .corpus import Corpus, MixedCorpus, load_corpus, save_corpus
from .model import LocusRegressor, tune_model
import argparse
from argparse import ArgumentTypeError
import os
import numpy as np
import sys
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def write_dataset(
        sep = '\t', 
        index = -1,*,
        fasta_file,
        genome_file,
        regions_file,
        vcf_files,
        exposure_files,
        correlates_files,
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

    if len(correlates_files) > 1 or len(exposure_files) > 1:
        
        assert len(correlates_files) == len(vcf_files),\
            'If more than one correlates or exposures file is provided, then the number of correlates files must match the number of VCF files.\n'\
            'E.g. each VCF file must be associated with a correlates file, in order.'

        assert len(exposure_files) in [0,1, len(vcf_files)],\
            'User must provide zero, one, or the number of exposure files which matches the number of VCF files.'

        dataset = MixedCorpus.create_corpus(
            **shared_args, 
            exposure_files = exposure_files,
            correlates_files = correlates_files,
        )
    
    else:

        dataset = Corpus.create_corpus(
            **shared_args,
            exposure_file = None if len(exposure_files) == 0 else exposure_files[0],
            correlates_file = correlates_files[0]
        )

    save_corpus(dataset, output)
    
    
def train_model(
        locus_subsample = 1,
        seed = 0, 
        pi_prior = 1.,
        num_epochs = 10000, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        quiet = True,
        n_jobs = 1,*,
        n_components,
        corpus,
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
    )
    
    dataset = load_corpus(corpus)
    
    model.fit(dataset)
    
    model.save(output)
    

def tune(
    locus_subsample = 0.1,
    pi_prior = 1.,
    num_epochs = 10000, 
    difference_tol = 1e-3,
    estep_iterations = 1000,
    bound_tol = 1e-2,
    n_jobs = 1,
    cv = 3,
    seed_reps = 1,
    max_candidates = 100,
    tune_pi_prior= False,
    factor = 2,
    target_epochs = 300,
    min_epochs = 30,
    num_tournaments = 4,*,
    output,
    corpus,
    min_components, 
    max_components,
):

    model = LocusRegressor(
        pi_prior = pi_prior,
        num_epochs = num_epochs, 
        difference_tol = difference_tol,
        estep_iterations = estep_iterations,
        bound_tol = bound_tol,
        n_jobs=1,
        locus_subsample=locus_subsample,
    )

    corpus = load_corpus(corpus)

    grid = tune_model(
        model, corpus,
        n_jobs = n_jobs,
        cv = cv,
        seed_reps=seed_reps,
        max_candidates = max_candidates,
        min_components = min_components,
        max_components = max_components,
        tune_pi_prior = tune_pi_prior,
        num_tournaments = num_tournaments,
        factor = factor,
        target_epochs = target_epochs,
        min_epochs=min_epochs,
    )

    print('Best params: ' + str(grid.best_params_))

    with open(output,'w') as f:
        print(*grid.cv_results_.keys(), file = f, sep = '\t')
        for row in zip(*grid.cv_results_.values()):
            print(*row, file = f, sep = '\t')
    

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers(help = 'commands')


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
dataset_sub.add_argument('--correlates-files', '-c', type = file_exists, nargs = '+', required=True,
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

trainer_sub = subparsers.add_parser('train-model', help = 'Train LocusRegression model on a pre-compiled corpus.')

trainer_required = trainer_sub.add_argument_group('Required arguments')
trainer_required .add_argument('--n-components','-k', type = posint, required=True,
    help = 'Number of signatures to learn.')
trainer_required .add_argument('--corpus', '-d', type = file_exists, required=True,
    help = 'Path to compiled corpus file.')
trainer_required .add_argument('--output','-o', type = valid_path, required=True,
    help = 'Where to save trained model.')

trainer_optional = trainer_sub.add_argument_group('Optional arguments')
trainer_optional .add_argument('--locus-subsample','-sub', type = posfloat, default = 1,
    help = 'Whether to use locus subsampling to speed up training via stochastic variational inference.')

trainer_optional.add_argument('--seed', type = posint, default=1776)
trainer_optional.add_argument('--pi-prior','-pi', type = posfloat, default = 1.,
    help = 'Dirichlet prior over sample mixture compositions. A value > 1 will give more dense compositions, which <1 finds more sparse compositions.')
trainer_optional.add_argument('--num-epochs', '-epochs', type = posint, default = 300,
    help = 'Maximum number of epochs to train.')
trainer_optional.add_argument('--bound-tol', '-tol', type = posfloat, default=1e-2,
    help = 'Early stop criterion, stop training if objective score does not increase by this much after one epoch.')
trainer_sub.set_defaults(func = train_model)



tune_sub = subparsers.add_parser('tune', help = 'Tune number of signatures for LocusRegression model on a pre-compiled corpus using the'
    'Successive Halving algorithm.')

tune_required = tune_sub.add_argument_group('Required arguments')
tune_required.add_argument('--corpus', '-d', type = file_exists, required=True,
    help = 'Path to compiled corpus file.')
tune_required.add_argument('--output','-o', type = valid_path, required=True,
    help = 'Where to save tuning results.')
tune_required.add_argument('--max-components','-max',type = posint, required=True,
    help= 'Maximum number of components to test for fit on dataset.')
tune_required.add_argument('--min-components','-min',type = posint, required=True,
    help= 'Maximum number of components to test for fit on dataset.')
tune_required.add_argument('--n-jobs','-j', type = posint, required= True,
    help = 'Number of concurrent jobs to run.')

tune_optional = tune_sub.add_argument_group('Optional arguments')

tune_optional.add_argument('--target-epochs', type = posint, default = 300,
    help = 'Number of epochs to train for on the last iteration of'
            ' successive halving. This should be set high enough such that the model converges to a solution.')
tune_optional.add_argument('--min-epochs', type = posint, default = 30,
    help = 'Minimum number of epochs to train before comparing models')

tune_optional.add_argument('--seed-reps', '-reps', type = posint, default=1,
    help = 'Number of different initialization seeds to try for each parameter combination.')

tune_optional.add_argument('--num-tournaments','-t', type = posint, default = 4,
    help = 'Number of hyperparameter tournaments to run.')
tune_optional.add_argument('--factor','-f',type = posint, default = 2,
    help = 'Successive halving reduction factor for each iteration')
tune_optional.add_argument('--max-candidates','-c', type = posint, default = 100,
    help = 'Maximum number of hyperparameter combinations to try.')
tune_optional.add_argument('--tune-pi-prior',action = 'store_true', default = False,
    help = 'Tune the dirichlet prior parameter alpha.')
tune_optional.add_argument('--cv','-cv', type = posint, default=3,
    help= 'Number of folds for cross-validation on training partition.')

model_options = tune_sub.add_argument_group('Model arguments')

model_options.add_argument('--locus-subsample','-sub', type = posfloat, default = 0.1,
    help = 'Whether to use locus subsampling to speed up training via stochastic variational inference.')
model_options.add_argument('--pi-prior','-pi', type = posfloat, default = 1.,
    help = 'Dirichlet prior over sample mixture compositions. A value > 1 will give more dense compositions, which <1 finds more sparse compositions.')
model_options.add_argument('--bound-tol', '-tol', type = posfloat, default=1e-2,
    help = 'Early stop criterion, stop training if objective score does not increase by this much after one epoch.')

tune_sub.set_defaults(func = tune)

def main():
    #____ Execute commands ___

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