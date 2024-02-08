import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import h5py as h5
import logging
from .sample import SampleLoader, InMemorySamples
from .sbs.observation_config import SBSSample
from tqdm import trange
logger = logging.getLogger('Corpus')


class CorpusMixin(ABC):

    def __init__(self,
        metadata = {},*,
        type,
        name,
        samples,
        features,
        context_frequencies,
        shared_exposures,
    ):
        self.type=type
        self.name = name
        self.samples = samples
        self.features = features
        self.context_frequencies = context_frequencies
        self._shared_exposures = shared_exposures
        self.metadata = metadata

        if self._shared_exposures:
            self._exposures = samples[0].exposures

    @abstractmethod
    def observation_class(self):
        raise NotImplementedError()

    @property
    def shape(self):
        return {
            'context_dim' : self.context_dim,
            'mutation_dim' : self.mutation_dim,
            'attribute_dim' : self.attribute_dim,
            'locus_dim' : self.locus_dim,
            'feature_dim' : self.feature_dim,
        }
    
    @property
    def feature_names(self):
        return list(self.features.keys())

    @property
    def exposures(self):
        return self._exposures

    @property
    def shared_correlates(self):
        return True
    
    @property
    def shared_exposures(self):
        return self._shared_exposures
        
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()

    @abstractmethod
    def subset_samples(self, idx):
        raise NotADirectoryError()

    @abstractmethod
    def subset_loci(self, loci):
        raise NotImplementedError()
    
    @abstractmethod
    def corpuses(self):
        raise NotImplementedError()
    
    @abstractmethod
    def corpus_names(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_corpus(self, name):
        raise NotImplementedError()


class Corpus(CorpusMixin):

    @classmethod
    def _get_observation_class(cls, classname):
        if classname.lower() == 'sbs':
            return SBSSample
        else:
            raise ValueError(f'Unknown corpus type {classname}')

    @property
    def observation_class(self):
        return self._get_observation_class(self.type)        
        
    @property
    def context_dim(self):
        return self.observation_class.N_CONTEXTS
    
    @property
    def mutation_dim(self):
        return self.observation_class.N_MUTATIONS
    
    @property
    def attribute_dim(self):
        return self.observation_class.N_ATTRIBUTES
    
    @property
    def locus_dim(self):
        return len(next(iter(self.features.values()))['values'])
    
    @property
    def feature_dim(self):
        return len(self.features)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample.corpus_name = self.name

        return sample


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    @property
    def corpuses(self):
        return [self]
    
    @property
    def corpus_names(self):
        return [self.name]
    
    @property
    def num_mutations(self):
        return sum([sum(sample.weight) for sample in self.samples])


    def subset_samples(self, subset_idx):

        return Corpus(
            type = self.type,
            samples = self.samples.subset(subset_idx),
            features=self.features,
            context_frequencies = self.context_frequencies,
            shared_exposures = self.shared_exposures,
            name = self.name,
        )


    def subset_loci(self, loci):

        subsample_lookup = dict(zip(loci, np.arange(len(loci)).astype(int)))
        
        bool_array = np.zeros(self.shape['locus_dim']).astype(bool)
        bool_array[loci] = True

        #total_mutations = 0
        new_samples = []
        for sample in self.samples:
                        
            mask = bool_array[sample.locus]

            new_sample = self.observation_class(**{
                'attribute' : sample.attribute[mask],
                'mutation' : sample.mutation[mask],
                'context' : sample.context[mask],
                'weight' : sample.weight[mask],
                'locus' : np.array([subsample_lookup[locus] for locus in sample.locus[mask]]).astype(int),
                'chrom' : sample.chrom[mask],
                'pos' : sample.pos[mask],
                'exposures' : sample.exposures[:,loci],   
                'name' : sample.name,
            })
            new_samples.append(new_sample)
        
        return Corpus(
            type = self.type,
            samples = InMemorySamples(new_samples),
            features = {
                feature_name : {
                    'type' : v['type'], 
                    'group' : v['group'],
                    'values' : v['values'][loci]
                }
                for feature_name, v in self.features.items()
            },
            context_frequencies = self.context_frequencies[:,loci],
            shared_exposures = self.shared_exposures,
            name = self.name,
        )
    

    def get_corpus(self, name):
        assert name == self.name
        return self
    

    def get_empirical_mutation_rate(self, use_weight=True):

        # returns the ln mutation rate for each locus in the first sample
        mutation_rate = self.samples[0].get_empirical_mutation_rate(use_weight = use_weight).toarray()

        # loop through the rest of the samples and add the mutation rate using logsumexp
        for i in trange(1, len(self), desc = 'Piling up mutations', ncols=100):
            mutation_rate = mutation_rate + self.samples[i].get_empirical_mutation_rate(use_weight = use_weight).toarray()

        return mutation_rate #.tocsr()



class MetaCorpus(CorpusMixin):
    
    def __init__(self, *corpuses):
        
        assert len(corpuses) > 1, 'If only one corpus, use that directly'
        assert len(set([corpus.name for corpus in corpuses])) == len(corpuses), \
            'All corpuses must have unique names.'
        assert all([np.all(corpuses[0].feature_names == corpus.feature_names) for corpus in corpuses])
        assert all([corpuses[0].shape == corpus.shape for corpus in corpuses])

        self._corpuses = corpuses
        self.idx_starts = np.cumsum(
            [0] + [len(corpus) for corpus in self.corpuses]
        )

    @property
    def observation_class(self):
        return self.corpuses[0].observation_class
    

    @property
    def corpuses(self):
        return self._corpuses

    @property
    def num_mutations(self):
        return sum([corpus.num_mutations for corpus in self.corpuses])

    @property
    def shape(self):
        return self.corpuses[0].shape
    
    @property
    def feature_dim(self):
        return self.corpuses[0].feature_dim
    
    @property
    def locus_dim(self):
        return self.corpuses[0].locus_dim
    
    @property
    def corpus_names(self):
        return [corpus.name for corpus in self.corpuses]


    def __len__(self):
        return sum([len(corpus) for corpus in self.corpuses])


    def _get_corpus_idx(self, idx):

        for corpus_idx, (idx_start, next_idx) in enumerate(zip(self.idx_starts, self.idx_starts[1:])):
            
            if idx >= idx_start and idx < next_idx:
                return corpus_idx, idx - idx_start

        raise IndexError()


    def __getitem__(self, idx):
    
        corpus_idx, sample_idx = self._get_corpus_idx(idx)
        return self.corpuses[corpus_idx][sample_idx]


    def __iter__(self):

        for corpus in self.corpuses:
            for sample in corpus:
                yield sample


    def subset_samples(self, subset_idx):
        
        corpus_idxs = defaultdict(list)
        for s in subset_idx:
            corp, idx = self._get_corpus_idx(s)
            corpus_idxs[corp].append(idx)

        return MetaCorpus(
            *[
                self.corpuses[i].subset_samples(idxs)
                for i, idxs in corpus_idxs.items()
            ]
        )


    def subset_loci(self, loci):
        
        return MetaCorpus(*[
            corpus.subset_loci(loci) for corpus in self.corpuses
        ])
    

    def get_corpus(self, name):
        
        for corpus in self.corpuses:
            if corpus.name == name:
                return corpus
            
        raise KeyError(f'Corpus {name} does not exist.')

    @property
    def shared_correlates(self):
        return False
    
    @property
    def context_frequencies(self):
        return self.corpuses[0].context_frequencies

    @property
    def feature_names(self):
        return self.corpuses[0].feature_names



def save_corpus(corpus, filename):

    with h5.File(filename, 'w') as f:
        
        data_group = f.create_group('data')
        data_group.attrs['shared_exposures'] = corpus.shared_exposures
        data_group.attrs['name'] = corpus.name
        data_group.attrs['type'] = corpus.type
        
        metadata_group = f.create_group('metadata')
        for key, val in corpus.metadata.items():
            metadata_group.attrs[key] = val

        data_group.create_dataset('context_frequencies', data = corpus.context_frequencies)
        features_group = data_group.create_group('features')

        for feature_name, feature in corpus.features.items():
            
            logger.debug(f'Saving correlate: {feature_name} ...')
            
            feature_group = features_group.create_group(feature_name)
            feature_group.attrs['type'] = feature['type']
            feature_group.attrs['group'] = feature['group']

            if np.issubdtype(feature['values'].dtype, np.str_):
                feature['values'] = feature['values'].astype('S')
            
            feature_group.create_dataset('values', data = feature['values'])

        samples_group = f.create_group('samples')

        for i, sample in enumerate(corpus.samples):
            sample.create_h5_dataset(samples_group, str(i))


def _peek_type(filename):
    
    with h5.File(filename, 'r') as f:
        return Corpus._get_observation_class(f['data'].attrs['type'])


def _read_features(data):

    features = {}
    feature_groups = data['data/features']
    
    for feature_name in feature_groups.keys():
        feature_group = feature_groups[feature_name]
        features[feature_name] = {
            'type' : feature_group.attrs['type'],
            'group' : feature_group.attrs['group'],
            'values' : feature_group['values'][...],
        }

    return features


def _load_corpus(filename, sample_obj):

    with h5.File(filename, 'r') as f:

        is_shared = f['data'].attrs['shared_exposures']

        if 'metadata' in f.keys():
            metadata = {
                key : val for key, val in f['metadata'].attrs.items()
            }
        else:
            metadata = {}


        return Corpus(
            type = f['data'].attrs['type'],
            context_frequencies = f['data/context_frequencies'][...],
            features = _read_features(f),
            samples = sample_obj,
            shared_exposures=is_shared,
            name = f['data'].attrs['name'],
            metadata=metadata
        )



def load_corpus(filename):

    sample_type = _peek_type(filename)

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
        SampleLoader(filename, observation_class=_peek_type(filename)
)
    )


def train_test_split(corpus, seed = 0, train_size = 0.7,
                     by_locus = False):

    randomstate = np.random.RandomState(seed)

    N = len(corpus) if not by_locus else corpus.locus_dim
    subset_fn = corpus.subset_samples if not by_locus else corpus.subset_loci

    train_idx = sorted(randomstate.choice(
                N, size = int(train_size * N), replace=False
            ))
    
    test_idx = sorted(list( set(range(N)).difference(set(train_idx)) )) 

    return subset_fn(train_idx), subset_fn(test_idx)