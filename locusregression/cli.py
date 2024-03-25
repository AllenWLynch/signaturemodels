from ._cli_utils import *
from .corpus import *
from .corpus.sbs.mutation_preprocessing import get_marginal_mutation_rate, get_mutation_clusters, transfer_annotations_to_vcf
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
import numpy as np
import tempfile
from contextlib import contextmanager
import time

from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers(help = 'Commands')


def get_marginal_mutation_rate_wrapper(*,
            vcf_files, 
            chr_prefix, 
            genome_file, 
            output,
            smoothing_size,
        ):
    
    get_marginal_mutation_rate(
        genome_file,
        output,
        *vcf_files,
        smoothing_size=smoothing_size,
        chr_prefix=chr_prefix
    )

prep_mutrate_parser = subparsers.add_parser('preprocess-estimate-mutrate', help = 'Calculate the marginal mutation rate across all samples for some corpus.')
prep_mutrate_parser.add_argument('--vcf-files', '-vcfs', type = file_exists, nargs='+', required=True, help = 'List of VCF files containing SBS mutations.')
prep_mutrate_parser.add_argument('--chr-prefix', default='', help = 'Prefix to append to chromosome names in VCF files.')
prep_mutrate_parser.add_argument('--smoothing-size', type = posint, default=20000, help = 'Size of window to smooth mutation rate over.')
prep_mutrate_parser.add_argument('--genome-file','-g', type = file_exists, required=True, help = 'Also known as a "Chrom sizes" file.')
prep_mutrate_parser.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout)
prep_mutrate_parser.set_defaults(func = get_marginal_mutation_rate_wrapper)



def get_mutation_clusters_wrapper(*,vcf_file, output, chr_prefix, sample, mutation_rate_bedgraph):
    
    mutations_df = get_mutation_clusters(
        mutation_rate_bedgraph=mutation_rate_bedgraph,
        vcf_file=vcf_file,
        chr_prefix=chr_prefix,
        sample=sample,
    )

    transfer_annotations_to_vcf(
        mutations_df,
        vcf_file=vcf_file,
        description = {
            'mutationType' : 'The type of mutation (e.g. C>A)',
            'negLog10interMutationDistanceRatio' : 'The negative log10 of the ratio of the inter-mutation distance to the critical distance',
            'clusterSize' : 'The number of mutations in the cluster',
            'cluster' : 'The mutation\'s cluster ID',
        },
        output = output,
        chr_prefix=chr_prefix,
    )

cluster_mutations_parser = subparsers.add_parser('preprocess-cluster-mutations', help = 'Cluster mutations in a VCF file.')
cluster_mutations_parser.add_argument('vcf-file', type = file_exists)
cluster_mutations_parser.add_argument('--mutation-rate-bedgraph','-m', type = file_exists, required=True, help = 'Bedgraph file of mutation rates.')
cluster_mutations_parser.add_argument('--output','-o', type = argparse.FileType('w'), help = 'Where to save clustered VCF file.')
cluster_mutations_parser.add_argument('--chr-prefix', default='', help = 'Prefix to append to chromosome names in VCF files.')
cluster_mutations_parser.add_argument('--sample','-s', type = str, default=sys.stdout, help = 'Sample name to filter mutations by.')
cluster_mutations_parser.set_defaults(func = get_mutation_clusters_wrapper)


def make_windows_wrapper(*,categorical_features, **kw):
    make_windows(*categorical_features, **kw)

make_windows_parser = subparsers.add_parser('get-regions', help = 'Make windows from a genome file.',
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
make_windows_parser.add_argument('--genome-file','-g', type = file_exists, required = True, help = 'Also known as a "Chrom sizes" file.')    
make_windows_parser.add_argument('--blacklist-file','-v', type = file_exists, required=True, help = 'Bed file of regions to exclude from windows.')
make_windows_parser.add_argument('--window-size','-w', type = posint, required = True, help = 'Size of windows to make.')
make_windows_parser.add_argument('--categorical-features','-cf', nargs='+', type = file_exists, default = [], 
                                 help = 'List of categorical feature bedfiles to account for while making windows.')
make_windows_parser.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout, 
                                 help = 'Where to save windows.')
make_windows_parser.set_defaults(func = make_windows_wrapper)


def create_corpus_wrapper(dtype='sbs',*,filename, corpus_name, fasta_file, regions_file):

    regions = read_windows(regions_file)

    SampleClass = Corpus._get_observation_class(dtype)

    context_frequencies = SampleClass.get_context_frequencies(
        window_set=regions,
        fasta_file=fasta_file,
    )

    create_corpus(
        filename=filename,
        name=corpus_name,
        type=dtype,
        context_frequencies=context_frequencies,
        regions=regions,
    )

corpus_init_parser = subparsers.add_parser('corpus-create', help = 'Create a corpus file from a fasta and regions file.')
corpus_init_parser.add_argument('filename', type = valid_path, help = 'Where to save compiled corpus.')
corpus_init_parser.add_argument('--corpus-name','-n', type = str, required = True, help = 'Name of corpus, must be unique if modeling with other corpuses.')
corpus_init_parser.add_argument('--fasta-file','-fa', type = file_exists, required = True, help = 'Sequence file, used to find context of mutations.')
corpus_init_parser.add_argument('--regions-file','-r', type = file_exists, required = True)
corpus_init_parser.add_argument('--dtype', type = str, default='sbs', choices=['sbs','fragment-motif','fragment-length','indel',])
corpus_init_parser.set_defaults(func = create_corpus_wrapper)


def list_features_wrapper(*,corpus):
    with h5.File(corpus, 'r') as f:
        if not 'features' in f or len(f['features']) == 0:
            print('No features found in corpus.', file = sys.stderr)
            os._exit(1)
            
        else:
            print('+' + '-' * 40 + '+' + '-' * 20 + '+' + '-' * 20 + '+')
            print('|{:^40s}|{:^20s}|{:^20s}|'.format('feature_name', 'group', 'normalization'))
            print('+' + '-' * 40 + '+' + '-' * 20 + '+' + '-' * 20 + '+')
            for feature in f['features'].keys():
                print('|{:^40s}|{:^20s}|{:^20s}|'.format(
                    feature,
                    f['features'][feature].attrs['group'],
                    f['features'][feature].attrs['type']
                ))
            print('+' + '-' * 40 + '+' + '-' * 20 + '+' + '-' * 20 + '+')
        
list_features_parser = subparsers.add_parser('corpus-list-features', help = 'List features in a corpus.')
list_features_parser.add_argument('corpus', type = file_exists)
list_features_parser.set_defaults(func = list_features_wrapper)

def remove_feature_wrapper(*,corpus, feature_names):
    with buffered_writer(corpus) as f:
        for feature_name in feature_names:
            try:
                del f['features'][feature_name]
            except KeyError:
                print(f'Feature {feature_name} not found in corpus.', file = sys.stderr)
                return

remove_feature_parser = subparsers.add_parser('corpus-rm-features', help = 'Remove a feature from a corpus.')
remove_feature_parser.add_argument('corpus', type = file_exists)
remove_feature_parser.add_argument('feature-names', type = str, nargs='+')
remove_feature_parser.set_defaults(func = remove_feature_wrapper)


def rename_corpus_wrapper(*,corpus,name):
    with buffered_writer(corpus) as f:
        data_group = f['metadata']
        data_group.attrs['name'] = name

rename_parser = subparsers.add_parser('corpus-rename', help = 'Rename a corpus.')
rename_parser.add_argument('corpus', type = file_exists)
rename_parser.add_argument('name', type = str)
rename_parser.set_defaults(func = rename_corpus_wrapper)


@contextmanager
def _get_regions_filename(corpus):

    try:
        regions_file = tempfile.NamedTemporaryFile()

        with h5.File(corpus, 'r') as h5_object, \
            open(regions_file.name, 'w') as regions_out:
            
            for region in read_regions(h5_object):
                print(region, sep = '\t', file = regions_out)

        yield regions_file.name
    finally:
        regions_file.close()


@contextmanager
def buffered_writer(filename, timeout=3600):
    init_time = time.time()
    opened=False
    
    while not opened:
        try:
            h5_object = h5.File(filename, 'a')
            opened = True
            yield h5_object
        except OSError:
            if time.time() - init_time > timeout:
                raise TimeoutError('Could not open file for writing.')
            else:
                time.sleep(1)
        finally:
            if opened:
                h5_object.close()
    



def process_bedgraph(
    group='all',
    normalization='power',
    extend=0,*,
    corpus,
    genome_file,
    bedgraph_file,
    feature_name,
):
    with _get_regions_filename(corpus) as regions_file:

        feature_vals = make_continous_features_bedgraph(
            bedgraph_file=bedgraph_file,
            regions_file=regions_file,
            genome_file=genome_file,
            extend=extend,
            null='nan',
        )

    with buffered_writer(corpus) as h5_object:
        write_feature(
            h5_object,
            name = feature_name,
            group = group,
            type = normalization,
            values = feature_vals,
        )

bedgraph_sub = subparsers.add_parser('corpus-ingest-bedgraph', help = 'Summarize bigwig file for a given cell type.')
bedgraph_sub.add_argument('corpus', type = file_exists, help = 'Path to compiled corpus file.')
bedgraph_sub.add_argument('bedgraph-file', type = file_exists)
bedgraph_sub.add_argument('--genome-file','-gf', type = file_exists, required=True)
bedgraph_sub.add_argument('--feature-name','-name', type = str, required=True,)
bedgraph_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
bedgraph_sub.add_argument('--extend','-e', type = posint, default=0, help = 'Extend each region by this many basepairs.')
bedgraph_sub.add_argument('--normalization','-norm', type = str, choices=['power','minmax','quantile','standardize'], 
                        default='power', 
                        help = 'Normalization to apply to feature.'
                        )
bedgraph_sub.set_defaults(func = process_bedgraph)


def process_vector(group='all',*,
                   normalization,
                   vector_file,
                   feature_name,
                   corpus):

    type_map = {
            'categorical' : str,
            'cardinality' : str,
        }

    feature_vals=[]
    with open(vector_file, 'r') as f:
        for line in f:
            _feature = line.strip()

            feature_vals.append(
                type_map.setdefault(normalization, float)(_feature) if not _feature=='.' else type_map.setdefault(normalization, float)('nan')
            )

    feature_vals = np.array(feature_vals)
    assert len(feature_vals) == peek_locus_dim(corpus)
    
    with buffered_writer(corpus) as h5_object:
        write_feature(
            h5_object,
            name = feature_name,
            group = group,
            type = normalization,
            values = feature_vals,
        )

vector_sub = subparsers.add_parser('corpus-ingest-vector', help = 'Ingest a vector file as a feature.')
vector_sub.add_argument('corpus', type = file_exists, help = 'Path to compiled corpus file.')
vector_sub.add_argument('vector-file', type = file_exists, help = 'A text file where each line is a feature value, should be the exact same length as the number of regions in the corpus')
vector_sub.add_argument('--feature-name','-name', type = str, required=True,)
vector_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
vector_sub.add_argument('--normalization','-norm', type = str, choices=['power','minmax','quantile','standardize','categorical','cardinality'], 
                        required=True,
                        help = 'Normalization to apply to feature.'
                        )
vector_sub.set_defaults(func = process_vector)


def process_bigwig(group='all',
                   normalization='power',
                   extend=0,*,
                   bigwig_file, 
                   feature_name, 
                   corpus):
    
    
    with _get_regions_filename(corpus) as regions_file:

        feature_vals = make_continous_features(
            bigwig_file=bigwig_file,
            regions_file=regions_file,
            extend=extend,
        )

    with buffered_writer(corpus) as h5_object:
        write_feature(
            h5_object,
            name = feature_name,
            group = group,
            type = normalization,
            values = feature_vals,
        )


bigwig_sub = subparsers.add_parser('corpus-ingest-bigwig', help = 'Summarize bigwig file for a given cell type.')
bigwig_sub.add_argument('corpus', type = file_exists, help = 'Path to compiled corpus file.')
bigwig_sub.add_argument('bigwig-file', type = file_exists)
bigwig_sub.add_argument('--feature-name','-name', type = str, required=True,)
bigwig_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
bigwig_sub.add_argument('--extend','-e', type = posint, default=0, help = 'Extend each region by this many basepairs.')
bigwig_sub.add_argument('--normalization','-norm', type = str, choices=['power','minmax','quantile','standardize'], 
                        default='power', 
                        help = 'Normalization to apply to feature.'
                        )
bigwig_sub.set_defaults(func = process_bigwig)


def process_distance_feature(
        group='all',
        normalization='quantile',
        reverse=False,*,
        bed_file,
        feature_name,
        corpus,
):
    
    with _get_regions_filename(corpus) as regions_file:
        progress_between, interdistance = make_distance_features(
            genomic_features=bed_file,
            reverse=reverse,
            regions_file=regions_file,
        )

    with buffered_writer(corpus) as h5_object:
        write_feature(
            h5_object,
            group = group,
            type = normalization,
            name = f'{feature_name}_progressBetween',
            values = progress_between,
        )

        write_feature(
            h5_object,
            group = group,
            type = normalization,
            name = f'{feature_name}_interFeatureDistance',
            values = interdistance,
        )

distance_sub = subparsers.add_parser('corpus-ingest-distance', help = 'Summarize distance to nearest feature upstream and downstream for some genomic elements.')
distance_sub.add_argument('corpus', type = file_exists, help = 'Path to compiled corpus file.')
distance_sub.add_argument('bed-file', type = file_exists, help = 'Bed file of genomic features. Only three columns are required, all other columns are ignored.')
distance_sub.add_argument('--feature-name','-name', type = str, required=True,)
distance_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
distance_sub.add_argument('--normalization','-norm', type = str, choices=['power','minmax','quantile','standardize'],
                            default='quantile', help = 'Normalization to apply to feature.')
distance_sub.set_defaults(func = process_distance_feature)


def process_discrete(
        group='all',*,
        bed_file,
        feature_name,
        corpus,
        null='.',
        class_priority=None,
        column=4,
        feature_type='categorical',
):

    with _get_regions_filename(corpus) as regions_file:
        discrete_features = make_discrete_features(
            genomic_features=bed_file,
            regions_file=regions_file,
            null=null,
            class_priority=class_priority,
            column=column,
        )
    
    with buffered_writer(corpus) as h5_object:
        write_feature(
            h5_object,
            name = feature_name,
            group = group,
            type = feature_type,
            values = discrete_features,
        )


discrete_sub = subparsers.add_parser('corpus-ingest-categorical', help = 'Summarize discrete genomic features for some genomic elements.')
discrete_sub.add_argument('corpus', type = file_exists, help = 'Path to compiled corpus file.')
discrete_sub.add_argument('bed-file', type = file_exists, help = 'Bed file of genomic features. Only three columns are required, all other columns are ignored.')
discrete_sub.add_argument('--feature-name','-name', type = str, required=True,)
discrete_sub.add_argument('--group','-g', type = str, default='all', help = 'Group name for feature.')
discrete_sub.add_argument('--null','-null', type = str, default='None', help = 'Value to use for missing features.')
discrete_sub.add_argument('--class-priority','-p', type = str, nargs = '+', default=None, help = 'Priority for resolving multiple classes for a single region.')
discrete_sub.add_argument('--column','-c', type = posint, default=4, help = 'Column in bed file to use for feature.')
discrete_sub.set_defaults(func = process_discrete)


cardinality_sub = subparsers.add_parser('corpus-ingest-cardinality', help = 'Summarize discrete genomic features for some genomic elements.')
cardinality_sub.add_argument('corpus', type = file_exists, help = 'Path to compiled corpus file.')
cardinality_sub.add_argument('bed-file', type = file_exists, help = 'Bed file of genomic features. The provided column must contain +/-/., all other columns are ignored.')
cardinality_sub.add_argument('--feature-name','-name', type = str, required=True,)
cardinality_sub.add_argument('--column','-c', type = posint, default=4, help = 'Column in bed file to use for feature.')
cardinality_sub.set_defaults(func = partial(process_discrete, 
                                         group = 'cardinality', 
                                         null = '.', 
                                         class_priority = ['+','-'],
                                         feature_type = 'cardinality',
                                        )
                            )

def ingest_sample(mutation_rate_bedgraph=None,
                  sample_name=None,
                  weight_col=None,
                  chr_prefix='',
                  *,corpus, sample_file, fasta_file
                ):
    
    if sample_name is None:
        sample_name = os.path.basename(sample_file)

    if mutation_rate_bedgraph is None:
        logger.warning('No mutation rate bedgraph provided, will not filter for clustered variants.')

    SampleClass = peek_type(corpus)

    with _get_regions_filename(corpus) as regions_file:
        sample = SampleClass.featurize_mutations(
            sample_file, 
            regions_file,
            fasta_file,
            chr_prefix = chr_prefix,
            weight_col = weight_col,
            mutation_rate_file = mutation_rate_bedgraph,
        )
    
    sample.name = sample_name

    with buffered_writer(corpus) as h5_object:
        write_sample(h5_object, sample, sample_name)

ingest_sample_parser = subparsers.add_parser('corpus-ingest-sample', help = 'Ingest a sample into a corpus.')
ingest_sample_parser.add_argument('corpus', type = file_exists, help = 'Path to compiled corpus file.')
ingest_sample_parser.add_argument('sample-file', type = file_exists, help = 'VCF file of mutations.')
ingest_sample_parser.add_argument('--sample-name','-name', type = str, default=None, help = 'Name of sample, defaults to filename.')
ingest_sample_parser.add_argument('--weight-col','-w', type = str, default=None, help = 'Column in INFO field to use as weight.')
ingest_sample_parser.add_argument('--mutation-rate-bedgraph','-m', type = file_exists, default=None, help = 'Bedgraph file of mutation rates.')
ingest_sample_parser.add_argument('--chr-prefix', default='', help = 'Prefix to append to chromosome names in VCF files.')
ingest_sample_parser.add_argument('--fasta-file','-fa', type = file_exists, required=True, help = 'Sequence file, used to find context of mutations.')
ingest_sample_parser.set_defaults(func = ingest_sample)


def list_samples(corpus):

    with h5.File(corpus, 'r') as f:
        if not 'samples' in f or len(f['samples']) == 0:
            print('No samples found in corpus.', file = sys.stderr)
            os._exit(1)

        else:
            for sample in f['samples'].keys():
                print(f['samples'][sample].attrs['name'], file = sys.stdout)

list_samples_parser = subparsers.add_parser('corpus-list-samples', help = 'List samples in a corpus.')
list_samples_parser.add_argument('corpus', type = file_exists)
list_samples_parser.set_defaults(func = list_samples)


def remove_sample(corpus, sample_names):

    with buffered_writer(corpus) as f:
        for sample_name in sample_names:
            try:
                del f['samples'][sample_name]
            except KeyError:
                print(f'Sample {sample_name} not found in corpus.', file = sys.stderr)

remove_sample_parser = subparsers.add_parser('corpus-rm-samples', help = 'Remove a sample from a corpus.')
remove_sample_parser.add_argument('corpus', type = file_exists)
remove_sample_parser.add_argument('sample-names', type = str, nargs='+')
remove_sample_parser.set_defaults(func = remove_sample)
    

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

    with _get_regions_filename(corpus) as regions_file:
        mutation_rate = stream_corpus(corpus)\
                        .get_empirical_mutation_rate()\
                        .sum((0,1))
        
        bed12_matrix_to_bedgraph(
            normalize_to_windowlength = True,
            regions_file = regions_file,
            matrix = mutation_rate[:,None],
            feature_names = ['mutation_rate'],
            output = output,
        )

empirical_mutrate_parser = subparsers.add_parser('corpus-empirical-mutrate',
    help = 'Aggregate mutations in a corpus to calculate the log (natural) empirical mutation rate. This depends on having sufficient mutations to find a smooth function.'
)
empirical_mutrate_parser.add_argument('corpus', type = file_exists)
empirical_mutrate_parser.add_argument('--output', '-o', type = argparse.FileType('w'), 
                                        default = sys.stdout)
empirical_mutrate_parser.set_defaults(func = empirical_mutation_rate)



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

    basemodel = get_basemodel(attrs["model_type"])
    
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

    basemodel = get_basemodel(model_type)
    
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

    dataset = load_dataset(corpuses)
    
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



def score(subset_by_loci=False,*,model, corpuses):

    dataset = load_dataset(corpuses)

    model = load_model(model)

    print(model.score(dataset, subset_by_loci=subset_by_loci))


score_parser = subparsers.add_parser('model-score', help='Score a model on a corpus.')
score_parser.add_argument('model', type = file_exists)
score_parser.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
    help = 'Path to compiled corpus file/files.')
score_parser.add_argument('--subset-by-loci', action = 'store_true', default=False)
score_parser.set_defaults(func = score)


def predict(*,model, corpuses, output):

    dataset = load_dataset(corpuses)

    model = load_model(model)

    exposures_matrix = model.predict_exposures(dataset)

    exposures_matrix.to_csv(output)

predict_sub = subparsers.add_parser('model-predict-exposures', help = 'Predict exposures for each sample in a corpus.')
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


def list_corpuses(*,model):
    model = load_model(model)
    print(*model.corpus_states.keys(), sep = '\n', file = sys.stdout)

list_corpuses_parser = subparsers.add_parser('model-list-corpuses',
                                             help = 'List the names of corpuses used to train a model.')
list_corpuses_parser.add_argument('model', type = file_exists)
list_corpuses_parser.set_defaults(func = list_corpuses)


def list_components(*,model):
    model = load_model(model)
    print(*model.component_names, sep = '\n', file = sys.stdout)

list_components_parser = subparsers.add_parser('model-list-components',
                                             help = 'List the names of corpuses used to train a model.')
list_components_parser.add_argument('model', type = file_exists)
list_components_parser.set_defaults(func = list_components)


def get_mutation_rate_r2(*, model, corpuses):

    model = load_model(model)
    dataset = load_dataset(corpuses)
    
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


def calc_locus_explanations(*,model,corpuses,components,n_jobs=1, subsample=10000):

    model_path = model
    model = load_model(model)
    dataset = load_dataset(corpuses)

    model.calc_locus_explanations(dataset,*components,n_jobs=n_jobs, subsample=subsample)
    model.save(model_path)

explain_parser = subparsers.add_parser('model-calc-explanations',
                                        help = 'Calculate the contribution of each feature to the mutation rate at each locus for each component.')
explain_parser.add_argument('model', type = file_exists)
explain_parser.add_argument('--corpuses', '-d', type = file_exists, nargs = '+', required=True,
                            help = 'Path to compiled corpus file/files.')
explain_parser.add_argument('--components', '-c', type = str, nargs = '+', default=[], required=False,
                            help = 'Components to calculate explanations for. If left unset, will calculate explanations for all components.')
explain_parser.add_argument('--n-jobs','-j', type = posint, default=1)
explain_parser.add_argument('--subsample','-s', type = posint, default=None)
explain_parser.set_defaults(func = calc_locus_explanations)


def get_mutation_rate_wrapper(*, model, corpus, output):

    model = load_model(model)

    with _get_regions_filename(corpus) as regions_file:

        logger.info('Loading corpus ...')
        corpus = stream_corpus(corpus)
        # Remove the effect for each context and direction by summing over 0,1 axes

        logger.info('Calculating marginal mutation rates ...')
        mr = np.exp(model.get_log_marginal_mutation_rate(corpus)).sum((0,1))

        logger.info('Writing mutation rates to bedgraph ...')
        bed12_matrix_to_bedgraph(
            normalize_to_windowlength=True,
            header=False,
            matrix = mr[:,None],
            regions_file = regions_file,
            feature_names=['mutation_rate'],
            output = output,
        )
    
mutrate_parser = subparsers.add_parser('model-get-marginal-mutrate',
                                        help = 'Get the marginal mutation rate for each region in a corpus.')
mutrate_parser.add_argument('model', type = file_exists)
mutrate_parser.add_argument('corpus', type = file_exists)
mutrate_parser.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout)
mutrate_parser.set_defaults(func = get_mutation_rate_wrapper)



def get_component_rate_wrapper(*, model, corpus, output):

    model = load_model(model)
    with _get_regions_filename(corpus) as regions_file:

        logger.info('Loading corpus ...')
        corpus = stream_corpus(corpus)

        logger.info('Calculating component mutation rates ...')
        component_rates = np.exp(model.get_log_component_mutation_rate(corpus))\
                                .sum((1,2)).T

        logger.info('Writing mutation rates to bedgraph ...')
        bed12_matrix_to_bedgraph(
            normalize_to_windowlength=True,
            header=True,
            matrix = component_rates,
            regions_file = regions_file,
            feature_names=model.component_names,
            output = output,
        )

component_rate_parser = subparsers.add_parser('model-get-component-mutrate',
                                        help = 'Get the mutation rate for each component in a corpus.')
component_rate_parser.add_argument('model', type = file_exists)
component_rate_parser.add_argument('corpus', type = file_exists)
component_rate_parser.add_argument('--output','-o', type = argparse.FileType('w'), default=sys.stdout)
component_rate_parser.set_defaults(func = get_component_rate_wrapper)


def _make_corpusstate_cache(*,model, corpus):

    cache_path = get_corpusstate_cache_path(model, corpus)

    model = load_model(model)
    corpus = stream_corpus(corpus)

    corpus_state = model._init_new_corpusstates(corpus)[corpus.name]

    joblib.dump(corpus_state, cache_path)

corpusstate_cache_parser = subparsers.add_parser('model-cache-sstats',
                                            help = 'Make a cache with pre-calculated statistcs for this model-corpus pair. This *greatly* speeds up subsequent calculations.')
corpusstate_cache_parser.add_argument('model', type = file_exists)
corpusstate_cache_parser.add_argument('--corpus','-d', type = file_exists, required=True)
corpusstate_cache_parser.set_defaults(func = _make_corpusstate_cache)



def _write_posterior_annotated_vcf(
    input_vcf, output_name,*,
    sample_class,
    model_state, 
    weight_col, 
    corpus_state,
    regions_file,
    fasta_file,
    chr_prefix,
    component_names,
    ):

    sample = sample_class.featurize_mutations(
                    input_vcf, 
                    regions_file,
                    fasta_file,
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
    
    with open(output_name, 'w') as output:
        transfer_annotations_to_vcf(
            posterior_df,
            vcf_file=input_vcf,
            description={}, 
            output=output, 
            chr_prefix=chr_prefix,
        )


def assign_components_wrapper(*,
        model, 
        vcf_files, 
        corpus, 
        output_prefix, 
        weight_col=None,
        regions_file,
        fasta_file,
        chr_prefix = '',
        n_jobs=1,
    ):

    SampleClass = peek_type(corpus)
    corpus_state = load_corpusstate_cache(model, corpus)
    
    model = load_model(model)
    
    with _get_regions_filename(corpus) as regions_file:

        annotation_fn = partial(
            _write_posterior_annotated_vcf,
            sample_class=SampleClass,
            model_state=model.model_state, 
            weight_col=weight_col, 
            corpus_state=corpus_state,
            regions_file=regions_file, 
            fasta_file= fasta_file,
            chr_prefix=chr_prefix,
            component_names=model.component_names,
        )

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
assign_components_parser.add_argument('--regions-file','-r', type = file_exists, required=True)
assign_components_parser.add_argument('--fasta-file','-fa', type = file_exists, required=True)
assign_components_parser.set_defaults(func = assign_components_wrapper)



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
                        seed=0, 
                        use_signatures=None,
                        n_jobs=1,
                        n_samples=None
                    ):
    
    corpus_state = load_corpusstate_cache(model, corpus)
    
    model = load_model(model)
    corpus = stream_corpus(corpus)

    if n_samples < len(corpus):
        assert n_samples > 0
        
        subset_idx = np.random.RandomState(seed).choice(len(corpus), size = n_samples, replace = False)
        corpus = corpus.subset_samples(subset_idx)

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
simulate_from_model_parser.add_argument('--n-samples','-samples', type = posint, default=None)
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

        try:
            func(**args)
        except BrokenPipeError:
            pass
