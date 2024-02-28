import numpy as np
from collections import defaultdict, Counter


def repeat_iterator(X, n):
    class reuseiterator:
        def __iter__(self):
            for i in range(n):
                yield X

    return reuseiterator()


class SampleSstats:

    def __init__(self,*,
        model_state,
        sample,
        weighted_phi,
        gamma,
    ):
        
        self.mutation_sstats =  np.zeros_like(model_state.rho_)
        self.context_sstats = np.zeros_like(model_state.lambda_)
        self.cardinality_sstats = np.zeros_like(model_state.tau_)
        
        self.locus_sstats = defaultdict(
                lambda : defaultdict(lambda : np.zeros(model_state.n_components))
            )
        

        for _cardinality, _mutation, _context, _locus, ss in zip(
            sample.cardinality, sample.mutation, sample.context, sample.locus, weighted_phi.T
        ):
            self.mutation_sstats[:, _context, _mutation]+=ss
            self.context_sstats[:, _context]+=ss
            self.locus_sstats[_locus][_cardinality] += ss

        self.gamma = gamma
        self.weighed_phi = weighted_phi


class CorpusSstats:

    def __init__(self, model_state):

        self.mutation_sstats =  np.zeros_like(model_state.rho_)
        self.context_sstats = np.zeros_like(model_state.lambda_)
        self.locus_sstats = defaultdict(
                lambda : defaultdict(lambda : np.zeros(model_state.n_components))
            )
        

        self._alpha_sstats = []
        self.exposures = []
        
        
    def __add__(self, sstats):

        self.mutation_sstats += sstats.mutation_sstats
        self.context_sstats += sstats.context_sstats
        self._alpha_sstats.append(sstats.gamma)

        for locus, stats in sstats.locus_sstats.items():
            for _cardinality, ss in stats.items():
                self.locus_sstats[locus][_cardinality] += ss

        return self
    

    def _convert_theta_sstats_to_array(self, k, n_bins):
        arr = np.zeros(n_bins)
        
        for l,v in self.locus_sstats.items():
            arr[l] = v[0][k] + v[1][k]
        return arr

    def theta_sstats(self, k, n_bins):
        return self._convert_theta_sstats_to_array(k, n_bins)
    

    def _convert_tau_sstats_to_array(self, k, n_bins):
        arr = np.zeros((2, n_bins))
        for l,v in self.locus_sstats.items():
            arr[0,l] = v[0][k]; arr[1,l] = v[1][k]
        return arr
    
    def tau_sstats(self, k, n_bins):
        return self._convert_tau_sstats_to_array(k, n_bins)
    

    def lambda_sstats(self, k):
        return self.context_sstats[k]
    
    def rho_sstats(self, k):
        return self.mutation_sstats[k]

    @property
    def alpha_sstats(self):
        return np.array(self._alpha_sstats)
    


class MetaSstats:
    '''
    De-nest and aggregate sstats across multiple corpuses. 
    * Mutation sstats (omega), context sstats (lambda) can be aggregated across all
    * For mutation rate estimation need to collect theta_sstats, exposures, and X_matrices

    Use the same interface as the CorpusSstats

    '''

    def __init__(self, corpus_sstats):

        first_corpus = list(corpus_sstats.values())[0]
        self.mutation_sstats =  np.zeros_like(first_corpus.mutation_sstats)
        self.context_sstats = np.zeros_like(first_corpus.context_sstats)

        for corpus in corpus_sstats.values():
            self.mutation_sstats += corpus.mutation_sstats
            self.context_sstats += corpus.context_sstats
            
        self.sstats = corpus_sstats


    def __getitem__(self, key):
        return self.sstats[key]
    

    def rho_sstats(self, k):
        return self.mutation_sstats[k]
            
