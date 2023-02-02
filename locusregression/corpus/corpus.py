import numpy as np
import pickle

def save_corpus(corpus, filename):

    with open(filename, 'wb') as f:
        pickle.dump(corpus, f)


def load_corpus(filename):

    with open(filename, 'rb') as f:
        return pickle.load(f)


class Corpus:

    def __init__(self,*,samples,
        window_size,
        feature_names,
        X_matrix,
        trinuc_distributions,
    ):
    
        self.samples = samples
        self.window_size = window_size
        self.feature_names = feature_names
        self.X_matrix = X_matrix
        self.trinuc_distributions = trinuc_distributions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        return {
            **self.samples[idx],
            'shared_correlates' : True,
            'window_size' : self.window_size, 
            'X_matrix' : self.X_matrix,
            'trinuc_distributions' : self.trinuc_distributions,
            'feature_names' : self.feature_names
        }

    def __iter__(self):

        for i in range(len(self)):
            yield self[i]


    def _partial_iter(self, subset_idx):

        class CorpusSubset:

            def __iter__(inside):
                for i in subset_idx:
                    yield self[i]

            def __len__(inside):
                return len(subset_idx)

        return CorpusSubset()


    def split_train_test(self, seed, train_size = 0.7):

        randomstate = np.random.RandomState(seed)

        train_idx = randomstate.choice(
            len(self), 
            size = int(train_size * len(self)),
            replace=False
        )

        test_idx = list( set( range(len(self)) ).difference(set(train_idx)) )

        return self._partial_iter(train_idx), self._partial_iter(test_idx)


class MixedCorpus(Corpus):
    
    def __getitem__(self, idx):
        
        return {
            **self.samples[idx],
            'shared_correlates' : False,
            'window_size' : self.window_size[idx], 
            'X_matrix' : self.X_matrix[idx],
            'trinuc_distributions' : self.trinuc_distributions,
            'feature_names' : self.feature_names
        }