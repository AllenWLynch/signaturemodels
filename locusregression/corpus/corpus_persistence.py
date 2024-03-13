import h5py as h5
from .corpus import Corpus
import numpy as np
import logging
from .corpus_maker import BED12Record
from .sample import InMemorySamples, SampleLoader
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def peek_type(filename):
    
    with h5.File(filename, 'r') as f:
        return Corpus._get_observation_class(f['metadata'].attrs['type'])
    

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
    
    for _id, region in enumerate(bed12_regions):
        region_container = regions_group.create_group(str(_id))
        for key in ['chromosome','start','end','name','block_count','block_sizes','block_starts']:
            region_container.attrs[key] = getattr(region, key)
        

def read_regions(h5_object):

    regions = []
    
    for _id in map(str, range(len(h5_object['regions'].keys()))):
        region = h5_object['regions'][_id]
        ## asserts _id is in order
        regions.append(
            BED12Record(
                chromosome = region.attrs['chromosome'],
                start = region.attrs['start'],
                end = region.attrs['end'],
                name = region.attrs['name'],
                score=0,
                strand='+',
                thick_end=0,
                thick_start=0,
                item_rgb='0,0,0',
                block_count= region.attrs['block_count'],
                block_sizes= region.attrs['block_sizes'],
                block_starts= region.attrs['block_starts'],
            )
        )

    return regions


def create_corpus(filename,*, name, type, context_frequencies, regions):

    with h5.File(filename, 'w') as f:
        data_group = f.create_group('metadata')
        data_group.attrs['name'] = name
        data_group.attrs['type'] = type

        f.create_dataset('context_frequencies', data = context_frequencies)

        write_regions(f, regions)



def save_corpus(corpus, filename):

    create_corpus(
                filename,
                name = corpus.name,
                type = corpus.type,
                context_frequencies = corpus.context_frequencies,
                regions = corpus.regions
            )
        
    with h5.File(filename, 'a') as f:    

        for feature_name, feature in corpus.features.items():
            write_feature(f, feature_name, feature)
            
        for i, sample in enumerate(corpus.samples):
            write_sample(f, sample, str(i))


def _load_corpus(filename, sample_obj):

    with h5.File(filename, 'r') as f:

        return Corpus(
            type = f['metadata'].attrs['type'],
            name = f['metadata'].attrs['name'],
            context_frequencies = f['context_frequencies'][...],
            features = read_features(f),
            samples = sample_obj,   
        )


def load_corpus(filename):

    sample_type = peek_type(filename)

    with h5.File(filename, 'r') as f:
        samples = InMemorySamples([
            sample_type.read_h5_dataset(f, f'samples/{i}')
            for i in range(len(f['samples'].keys()))
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


