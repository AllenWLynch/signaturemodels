import h5py as h5
from .corpus import Corpus
import numpy as np
import logging
from .reader_utils import BED12Record
from .sample import InMemorySamples, SampleLoader
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def peek_type(filename):
    
    with h5.File(filename, 'r') as f:
        return Corpus._get_observation_class(f['metadata'].attrs['type'])
    

def peek_locus_dim(filename):    
    with h5.File(filename, 'r') as f:
        return len(f['regions']['chromosome'][...])
        

def peek_type_opened(h5_object):
    return Corpus._get_observation_class(h5_object['metadata'].attrs['type'])


def write_feature(h5_object,*, name, group, type, values):

    logger.debug(f'Saving correlate: {name} ...')

    if not 'features' in h5_object:
        h5_object.create_group('features')
    
    features_group = h5_object['features']
    
    if name in features_group:
        del features_group[name]

    feature_group = features_group.create_group(name)
    feature_group.attrs['type'] = type
    feature_group.attrs['group'] = group

    if np.issubdtype(values.dtype, np.str_):
        values = values.astype('S')
    
    feature_group.create_dataset('values', data = values)


def read_feature(h5_object, feature_name):

    feature_group = h5_object['features'][feature_name]

    vals = feature_group['values'][...]

    if vals.dtype.kind == 'S':
        vals = np.char.decode(vals, 'utf-8')

    return {
        'type' : feature_group.attrs['type'],
        'group' : feature_group.attrs['group'],
        'values' : vals,
    }
    

def delete_feature(h5_object, feature_name):
    del h5_object['features'][feature_name]
    

def read_features(h5_object):

    features = {}    
    for feature_name in h5_object['features'].keys():
        features[feature_name] = read_feature(h5_object, feature_name)

    return features


def write_sample(h5_object, sample, sample_name):
    if not 'samples' in h5_object:
        h5_object.create_group('samples')

    if sample_name in h5_object['samples']:
        del h5_object['samples'][sample_name]
        
    sample.create_h5_dataset(h5_object['samples'], sample_name)


def read_sample(h5_object, sample_name):
    observation_type = peek_type_opened(h5_object)
    return observation_type.read_h5_dataset(h5_object['samples'], sample_name)


def delete_sample(h5_object, sample_name):
    del h5_object['samples'][sample_name]


def write_regions(h5_object, bed12_regions):

    if not 'regions' in h5_object:
        h5_object.create_group('regions')
    regions_group = h5_object['regions']
        
    save_attrs = ['chromosome','start','end','name','block_count','block_sizes','block_starts']
    for attribute in save_attrs:
        vals = [getattr(region, attribute) for region in bed12_regions]
        if isinstance(vals[0], str):
            vals = np.array(vals).astype('S')
        elif type(vals[0]) == list:
            vals = np.array(list(map(lambda x : ','.join(map(str, x)), vals))).astype('S')
        else:
            vals = np.array(vals)
        
        regions_group.create_dataset(attribute, data = vals)


def read_regions(h5_object):

    regions = []
    read_attrs = ['chromosome','start','end','name','block_count','block_sizes','block_starts']

    def convert_dtype(key, val):
        if val.dtype.kind == 'S':
            val = list(np.char.decode(val, 'utf-8'))

        if key in ['block_sizes', 'block_starts']:
            val = list(
                map(lambda x : list(map(int, x.split(','))), val)
            )
        
        return val
        
    data = {key: convert_dtype(key, h5_object['regions'][key][...]) for key in read_attrs}
    
    for i in range(len(data['chromosome'])):
        region_dict = {key: data[key][i] for key in read_attrs}
        ## asserts _id is in order
        regions.append(
            BED12Record(
                **region_dict,
                score=0,
                strand='+',
                thick_end=0,
                thick_start=0,
                item_rgb='0,0,0',
            )
        )

    return regions


def create_corpus(filename, exposures=None,*,name, type, context_frequencies, regions):

    with h5.File(filename, 'w') as f:
        data_group = f.create_group('metadata')
        data_group.attrs['name'] = name
        data_group.attrs['type'] = type

        f.create_dataset('context_frequencies', data = context_frequencies)
        
        if exposures is None:
            exposures = np.ones((1,context_frequencies.shape[-1]))

        f.create_dataset('exposures', data = exposures)

        write_regions(f, regions)



def save_corpus(corpus, filename):

    create_corpus(
                filename,
                name = corpus.name,
                type = corpus.type,
                context_frequencies = corpus.context_frequencies,
                regions = corpus.regions,
                exposures = corpus.exposures,
            )
        
    with h5.File(filename, 'a') as f:    

        for feature_name, feature in corpus.features.items():
            write_feature(f, name=feature_name, **feature)
            
        for i, sample in enumerate(corpus.samples):
            write_sample(f, sample, sample.name)


def _load_corpus(filename, sample_obj):

    with h5.File(filename, 'r') as f:

        return Corpus(
            type = f['metadata'].attrs['type'],
            name = f['metadata'].attrs['name'],
            context_frequencies = f['context_frequencies'][...],
            features = read_features(f),
            regions = read_regions(f),
            exposures = f['exposures'][...],
            samples = sample_obj,
        )


def load_corpus(filename):

    sample_type = peek_type(filename)

    with h5.File(filename, 'r') as f:
        samples = InMemorySamples([
            sample_type.read_h5_dataset(f, f'samples/{i}')
            for i in f['samples'].keys()
        ])

    return _load_corpus(
        filename,
        samples,
    )
    

def stream_corpus(filename):

    return _load_corpus(
        filename,
        SampleLoader(filename, observation_class=peek_type(filename))
    )


