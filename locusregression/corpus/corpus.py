import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import h5py as h5
import logging
from .sbs_sample import SBSSample
logger = logging.getLogger('Corpus')

class CorpusMixin(ABC):

    def __init__(self,
        metadata = {},*,
        name,
        samples,
        feature_names,
        X_matrix,
        trinuc_distributions,
        shared_exposures,
    ):
        self.name = name
        self.samples = samples
        self.feature_names = feature_names
        self.X_matrix = X_matrix
        self.trinuc_distributions = trinuc_distributions
        self._shared_exposures = shared_exposures
        self.metadata = metadata

        if self._shared_exposures:
            self._exposures = samples[0].exposures

    @property
    def shape(self):
        return self.X_matrix.shape

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
    def get_corpus(self, name):
        raise NotImplementedError()



class SampleLoader:

    def __init__(self, filename, subset_idx = None):
        
        self.filename = filename
        
        if subset_idx is None:
            
            with h5.File(self.filename, 'r') as f:
                n_samples = len(f['samples'].keys())

            subset_idx = list(range(n_samples))
        
        self.subset_idx = subset_idx


    def __len__(self):
        return len(self.subset_idx)


    def _read_item(self, h5, idx):
        return SBSSample.read_h5_dataset(h5, f'samples/{idx}')


    def __iter__(self):

        with h5.File(self.filename, 'r') as f:
            for i in self.subset_idx:
                yield self._read_item(f,i)
        

    def __getitem__(self, idx):
        
        idx = self.subset_idx[idx]

        with h5.File(self.filename, 'r') as f:
            return self._read_item(f, idx)


    def subset(self, idx_list):
        return SampleLoader(self.filename, [self.subset_idx[i] for i in idx_list])


class InMemorySamples(list):

    def subset(self, idx_list):
        return InMemorySamples([self[i] for i in idx_list])


def save_corpus(corpus, filename):

    with h5.File(filename, 'w') as f:
        
        data_group = f.create_group('data')
        data_group.attrs['shared_exposures'] = corpus.shared_exposures
        data_group.attrs['name'] = corpus.name
        
        metadata_group = f.create_group('metadata')
        for key, val in corpus.metadata.items():
            metadata_group.attrs[key] = val

        data_group.create_dataset('trinuc_distributions', data = corpus.trinuc_distributions)
        data_group.create_dataset('X_matrix', data = corpus.X_matrix)
        data_group['X_matrix'].attrs['feature_names'] = corpus.feature_names

        samples_group = f.create_group('samples')

        for i, sample in enumerate(corpus.samples):
            sample.create_h5_dataset(samples_group, str(i))


def overwrite_corpus_features(filename, X_matrix, feature_names):

    n_features, n_loci_X = X_matrix.shape
    assert len(feature_names) == n_features, 'The number of feature names provided must match the first dimension of the provided X_matrix.'

    with h5.File(filename, 'a') as f:

        _, n_loci = f['data/trinuc_distributions'].shape

        assert n_loci_X == n_loci, \
            f'The new provided feature matrix does not have the required number of loci (Expected {n_loci}, got {n_loci_X}).'

        data_group = f['data']

        old_feature_names = data_group['X_matrix'].attrs['feature_names']
        old_features = data_group['X_matrix'][...]

        try:
            del data_group['X_matrix']
            data_group.create_dataset('X_matrix', data = X_matrix)
            data_group['X_matrix'].attrs['feature_names'] = feature_names

        except Exception as err:
            
            if 'X_matrix' in data_group.keys():
                del data_group['X_matrix']

            data_group.create_dataset('X_matrix', data = old_features)
            data_group['X_matrix'].attrs['feature_names'] = old_feature_names
            raise err
        


def load_corpus(filename):

    with h5.File(filename, 'r') as f:

        is_shared = f['data'].attrs['shared_exposures']

        if 'metadata' in f.keys():
            metadata = {
                key : val for key, val in f['metadata'].attrs.items()
            }
        else:
            metadata = {}

        

        return Corpus(
            trinuc_distributions = f['data/trinuc_distributions'][...],
            X_matrix = f['data/X_matrix'][...],
            feature_names = f['data/X_matrix'].attrs['feature_names'],
            samples = InMemorySamples([
                SBSSample.read_h5_dataset(f, f'samples/{i}')
                for i in range(len(f['samples'].keys()))
            ]),
            shared_exposures=is_shared,
            name = f['data'].attrs['name'],
            metadata=metadata
        )


def stream_corpus(filename):

    with h5.File(filename, 'r') as f:

        is_shared = f['data'].attrs['shared_exposures']

        if 'metadata' in f.keys():
            metadata = {
                key : val for key, val in f['metadata'].attrs.items()
            }
        else:
            metadata = {}
            

        return Corpus(
            trinuc_distributions = f['data/trinuc_distributions'][...],
            X_matrix = f['data/X_matrix'][...],
            feature_names = f['data/X_matrix'].attrs['feature_names'],
            samples = SampleLoader(filename),
            shared_exposures=is_shared,
            name = f['data'].attrs['name'],
            metadata=metadata
        )


def train_test_split(corpus, seed = 0, train_size = 0.7):

    randomstate = np.random.RandomState(seed)

    train_idx = sorted(
        randomstate.choice(
            len(corpus), 
            size = int(train_size * len(corpus)),
            replace=False
        )
    )

    test_idx = sorted( list( set( range(len(corpus)) ).difference(set(train_idx)) ) )

    return corpus.subset_samples(train_idx), corpus.subset_samples(test_idx)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Corpus(CorpusMixin):

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        
        return dotdict({
            **self.samples[idx].asdict(),
            'corpus_name' : self.name
        })


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    @property
    def corpuses(self):
        return [self]
    
    @property
    def num_mutations(self):
        return sum([sum(sample.weights) for sample in self.samples])


    def subset_samples(self, subset_idx):

        return Corpus(
            samples = self.samples.subset(subset_idx),
            feature_names = self.feature_names,
            X_matrix = self.X_matrix,
            trinuc_distributions = self.trinuc_distributions,
            shared_exposures = self.shared_exposures,
            name = self.name,
            metadata = self.metadata,
        )


    def subset_loci(self, loci):

        subsample_lookup = dict(zip(loci, np.arange(len(loci)).astype(int)))
        
        bool_array = np.zeros(self.X_matrix.shape[1]).astype(bool)
        bool_array[loci] = True

        #total_mutations = 0
        new_samples = []
        for sample in self.samples:
                        
            mask = bool_array[sample.locus]

            new_sample = SBSSample(**{
                'mutation' : sample.mutation[mask],
                'context' : sample.context[mask],
                'weight' : sample.weight[mask],
                'locus' : np.array([subsample_lookup[locus] for locus in sample.locus[mask]]).astype(int), 
                'exposures' : sample.exposures[:,loci],   
                'name' : sample.name,
            })

            #total_mutations += new_sample.count.sum()

            new_samples.append(new_sample)
        
        return Corpus(
            samples = InMemorySamples(new_samples),
            X_matrix=self.X_matrix[:,loci],
            trinuc_distributions = self.trinuc_distributions[:,loci],
            feature_names = self.feature_names,
            shared_exposures = self.shared_exposures,
            name = self.name,
        )
    

    def get_corpus(self, name):
        assert name == self.name
        return self
    

    def collapse_mutations(self):
        
        all_mutations = defaultdict(lambda : 0)

        for sample in self.samples:

            for (mut, context, locus, weight) in \
                zip(sample.mutation, sample.context, sample.locus, sample.weight):
                
                all_mutations[(mut, context, locus)] += weight

        mutation, context, locus = list(zip(*all_mutations.keys()))
                                        
        new_sample = SBSSample(**dict(
            mutation = np.array(mutation),
            context = np.array(context),
            locus = np.array(locus),
            weight = np.array(list(all_mutations.values())),
            exposures = self.exposures,
            name = self.name,
        ))

        return Corpus(
            samples = InMemorySamples([new_sample]),
            X_matrix=self.X_matrix,
            trinuc_distributions = self.trinuc_distributions,
            feature_names = self.feature_names,
            shared_exposures = self.shared_exposures,
            name = self.name,
        )


    def get_empirical_mutation_rate(self, use_weight=True):

        # returns the ln mutation rate for each locus in the first sample
        mutation_rate = self.samples[0].get_empirical_mutation_rate(use_weight = use_weight)

        # loop through the rest of the samples and add the mutation rate using logsumexp
        for i in range(1, len(self)):
            mutation_rate = mutation_rate + self.samples[i].get_empirical_mutation_rate()

        return mutation_rate



class MetaCorpus(CorpusMixin):
    
    def __init__(self, *corpuses):
        
        assert len(corpuses) > 1, 'If only one corpus, use that directly'
        assert len(set([corpus.name for corpus in corpuses])) == len(corpuses), \
            'All corpuses must have unique names.'
        assert all([np.all(corpuses[0].feature_names == corpus.feature_names) for corpus in corpuses])
        assert all([corpuses[0].X_matrix.shape == corpus.X_matrix.shape for corpus in corpuses])

        self._corpuses = corpuses
        self.idx_starts = np.cumsum(
            [0] + [len(corpus) for corpus in self.corpuses]
        )

    @property
    def corpuses(self):
        return self._corpuses

    @property
    def num_mutations(self):
        return sum([corpus.num_mutations for corpus in self.corpuses])

    @property
    def shape(self):
        return self.corpuses[0].X_matrix.shape


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
    def trinuc_distributions(self):
        return self.corpuses[0].trinuc_distributions

    @property
    def feature_names(self):
        return self.corpuses[0].feature_names
