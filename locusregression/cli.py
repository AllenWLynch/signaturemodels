from locusregression import Corpus, MixedCorpus, load_corpus, LocusRegressor
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
        ):
    pass

def write_dataset(
        sep = '\t', 
        index = -1,*,
        fasta_file,
        genome_file,
        regions_file,
        vcf_files,
        exposure_files,
        correlates_files,
        save_name
        ):

    shared_args = dict(
        fasta_file = fasta_file, 
        genome_file = genome_file,
        regions_file = regions_file,
        vcf_files = vcf_files,
        sep = sep, index = index,
    )

    if len(correlates_files) > 1 or len(exposure_files) > 1:
        
        assert len(correlates_files) == len(vcf_files),\
            'If more than one correlates or exposures file is provided, then the number of correlates files must match the number of VCF files.\n'\
            'E.g. each VCF file must be associated with a correlates file, in order.'

        assert len(exposure_files) in [0,1, len(vcf_files)],\
            'User must provide zero, one, or the number of exposure files which matches the number of VCF files.'


        dataset = MixedCorpus(
            **shared_args, 
            exposure_files = exposure_files,
            correlates_files = correlates_files,
        )
    
    else:

        dataset = Corpus(
            **shared_args,
            exposure_file = None if len(exposure_files) == 0 else exposure_files[0],
            correlates_file = correlates_files[0]
        )

    dataset.save(save_name)
    
    
def train_model(
        seed = 0, 
        pi_prior = 5,
        num_epochs = 10000, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        quiet = True,*,
        n_components,
        dataset,
        save_name,
    ):
    
    model = LocusRegressor(
        seed = seed, 
        dtype = np.float32,
        pi_prior = pi_prior,
        num_epochs = num_epochs, 
        difference_tol = difference_tol,
        estep_iterations = estep_iterations,
        bound_tol = bound_tol,
        quiet = quiet,
        n_components = n_components
    )
    
    dataset = load_corpus(dataset)
    
    model.fit(dataset)
    
    model.save(save_name)
    

    
def evaluate_model(*,
        model_file,
        dataset_file
    ):
    
    model = LocusRegressor.load(model_file)
    
    dataset = load_corpus(dataset_file)
    
    print(
        model.score(dataset)
    )
    
    
class MyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

parser = MyArgumentParser(
    fromfile_prefix_chars = '@',
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
    

dataset_sub = subparsers.add_parser('write-dataset')
dataset_sub.add_argument('--vcf-files', '-vcf', nargs = '+', type = file_exists, required = True)
dataset_sub.add_argument('--sep','-sep',default ='\t',type = str)
dataset_sub.add_argument('--index','-i',default = -1,type = int, 
                         help = 'Position offset between VCF and Fasta/Bed files. If VCF files were written with 1-based indexing,'
                                'set this to "-1".')
dataset_sub.add_argument('--fasta-file','-fa', type = file_exists, required = True)
dataset_sub.add_argument('--genome-file','-g', type = file_exists, required = True)
dataset_sub.add_argument('--regions-file','-r', type = file_exists, required = True)
dataset_sub.add_argument('--correlates-files', '-c', type = file_exists, nargs = '+', required=True)
dataset_sub.add_argument('--exposure-files','-e', type = file_exists, nargs = '+')

dataset_sub.add_argument('--save-name','-f', type = valid_path, required = True)
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

trainer_sub = subparsers.add_parser('train')
trainer_sub.add_argument('--n-components','-k', type = posint, required=True)
trainer_sub.add_argument('--dataset', '-d', type = file_exists, required=True)
trainer_sub.add_argument('--save-name', type = valid_path, required=True)

trainer_sub.add_argument('--seed', type = posint, default=1776)
trainer_sub.add_argument('--pi-prior','-pi', type = posfloat, default = 1.)
trainer_sub.add_argument('--num-epochs', type = posint, default = 300)
trainer_sub.add_argument('--bound-tol', '-tol', type = posfloat, default=1e-2)
trainer_sub.set_defaults(func = train_model)


def main():
    #____ Execute commands ___

    args = parser.parse_args()

    try:
        args.func #first try accessing the .func attribute, which is empty if user tries ">>>lisa". In this case, don't throw error, display help!
    except AttributeError:
        print(parser.print_help(), file = sys.stderr)
    else:
        
        args = vars(args)
        func = args.pop('func')
        args = {k.replace('-','_') : v for k,v in args.items()}

        func(**args)