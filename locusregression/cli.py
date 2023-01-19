from . import Dataset
from . import LocusRegressor
import argparse
import os

def write_dataset(
        sep = '\t', 
        index = -1,*,
        fasta_file,
        genome_file,
        vcf_files,
        bedgraph_matrix,
        save_name
        ):

    dataset = Dataset(
            fasta_file = fasta_file, 
            genome_file = genome_file,
            bedgraph_mtx = bedgraph_matrix,
            vcf_files = vcf_files,
           )

    dataset.save(save_name)
    
    
def train_model(
        seed = 0, 
        eval_every = 10,
        pi_prior = 5,
        num_epochs = 10000, 
        difference_tol = 1e-3,
        estep_iterations = 1000,
        bound_tol = 1e-2,
        quiet = True,*,
        num_components,
        dataset_file,
        save_name,
    ):
    
    model = LocusRegressor(
        seed = seed, 
        eval_every = eval_every,
        dtype = dtype,
        pi_prior = pi_prior,
        num_epochs = num_epochs, 
        difference_tol = difference_tol,
        estep_iterations = estep_iterations,
        bound_tol = bounds_tol,
        quiet = quiet,
        num_components = num_components
    )
    
    dataset = Dataset.load(dataset_file)
    
    model.fit(dataset)
    
    model.save(save_name)
    
    
    
def evaluate_model(*,
        model_file,
        dataset_file
    ):
    
    model = LocusRegressor.load(model_file)
    
    dataset = Dataset.load(dataset_file)
    
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
        raise ValueError('File {} does not exist.'.format(x))

def valid_path(x):
    if not os.access(x, os.W_OK):
        
        try:
            open(x, 'w').close()
            os.unlink(x)
            return x
        except OSError as err:
            raise ValueError('File {} cannot be written. Invalid path.'.format(x)) from err
    
    return x
    

dataset_sub = subparsers.add_parser('write-dataset')
dataset_sub.add_argument('vcf-files', nargs = '+', type = file_exists)
dataset_sub.add_argument('--sep','-sep',default ='\t',type = str)
dataset_sub.add_argument('--index','-i',default = -1,type = int, 
                         help = 'Position offset between VCF and Fasta/Bed files. If VCF files were written with 1-based indexing,'
                                'set this to "-1".')
dataset_sub.add_argument('--fasta-file','-fa', type = file_exists, required = True)
dataset_sub.add_argument('--genome-file','-g', type = file_exists, required = True)
dataset_sub.add_argument('--bedgraph-matrix','-b', type = file_exists, required = True)
dataset_sub.add_argument('--save-name','-f', type = valid_path, required = True)
dataset_sub.set_defaults(func = write_dataset)


def main():
    #____ Execute commands ___

    args = parser.parse_args()

    try:
        args.func #first try accessing the .func attribute, which is empty if user tries ">>>lisa". In this case, don't throw error, display help!
    except AttributeError:
        print(parser.print_help(), file = sys.stderr)
    else:
        
        args = vars(args)
        args.pop('func')
        
        args.func(**args)