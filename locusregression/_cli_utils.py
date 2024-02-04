
from .model import GBTRegressor, LocusRegressor
from .corpus import stream_corpus, MetaCorpus
from tempfile import NamedTemporaryFile
from argparse import ArgumentTypeError
import subprocess
import os
import joblib

def get_corpusstate_cache_path(model_path, corpus_path):
    return os.path.join(
        os.path.dirname(corpus_path),
        '.' + os.path.basename(corpus_path) + os.path.relpath(model_path, corpus_path)\
            .replace('/','.')\
            .replace(' ','') \
        + '.corpusstate'
    )


def load_corpusstate_cache(model_path, corpus_path):
    
    cache_path = get_corpusstate_cache_path(model_path, corpus_path)

    if not os.path.exists(cache_path):
        print(cache_path)
        raise FileNotFoundError(
            f'A corpus state cache was not found for this model, corpus pairing: {model_path}, {corpus_path}.\n'
            'You can generate on by using the following command:\n'
            f'\t$ locusregression model-cache-sstats {model_path} -d {corpus_path}'
        )

    model_mtime = os.path.getmtime(model_path)
    corpus_mtime = os.path.getmtime(corpus_path)
    cache_mtime = os.path.getmtime(cache_path)

    assert model_mtime > corpus_mtime, 'Model must have been trained after corpus was last modified.'
    assert cache_mtime > model_mtime, 'Cache must have been created after model was trained'

    # check if model was saved after corpus, and if path was saved after model
    return joblib.load(cache_path)


def transfer_annotations_to_vcf(
        annotations_df,*,
        vcf_file,
        description, 
        output, 
        chr_prefix=''
    ):

    annotations_df = annotations_df.copy()

    assert 'CHROM' in annotations_df.columns, 'Annotations must have a column named "CHROM".'
    assert 'POS' in annotations_df.columns, 'Annotations must have a column named "POS".'

    annotations_df['CHROM'] = annotations_df.CHROM.str.removeprefix(chr_prefix)
    annotations_df['POS'] = annotations_df.POS + 1 #switch to 1-based from 0-based indexing
    annotations_df = annotations_df.sort_values(['CHROM','POS'])

    transfer_columns = ','.join(['CHROM','POS'] + ['INFO/' + c for c in annotations_df.columns if not c in ['CHROM','POS']])

    with NamedTemporaryFile() as header, \
        NamedTemporaryFile(delete=False) as dataframe:

        with open(header.name, 'w') as f:
            for col in annotations_df.columns:
                
                if col in ['CHROM','POS']:
                    continue
                
                dtype = str(annotations_df[col].dtype)
                if dtype.startswith('int'):
                    dtype = 'Integer'
                elif dtype.startswith('float'):
                    dtype = 'Float'
                else:
                    dtype = 'String'

                print(
                    f'##INFO=<ID={col},Number=1,Type={dtype},Description="{description}">',
                    file = f,
                    sep = '\n',
                )

            annotations_df.to_csv(dataframe.name, index = None, sep = '\t', header = None)

        try:    
            subprocess.check_output(['bgzip','-f',dataframe.name])
            subprocess.check_output(['tabix','-s1','-b2','-e2', '-f', dataframe.name + '.gz'])

            subprocess.check_output(
                ['bcftools','annotate',
                '-a',  dataframe.name + '.gz',
                '-h', header.name,
                '-c', transfer_columns,
                '-o', output,
                vcf_file,
                ]
            )
        
        finally:
            os.remove(dataframe.name + '.gz')
            os.remove(dataframe.name + '.gz.tbi')


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


def load_dataset(corpuses):

    if len(corpuses) == 1:
        dataset = stream_corpus(corpuses[0])
    else:
        dataset = MetaCorpus(*[
            stream_corpus(corpus) for corpus in corpuses
        ])

    return dataset


def get_basemodel(model_type):

    if model_type == 'linear':
        basemodel = LocusRegressor
    elif model_type == 'gbt':
        basemodel = GBTRegressor
    else:
        raise ValueError(f'Unknown model type {model_type}')

    return basemodel

