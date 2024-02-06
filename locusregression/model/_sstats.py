import numpy as np
from collections import defaultdict


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
        
        self.mutation_sstats =  np.zeros_like(model_state.omega)
        self.context_sstats = np.zeros_like(model_state.delta)
        
        self.locus_sstats = defaultdict(lambda : np.zeros(model_state.n_components))

        for _mutation, _context, _locus, ss in zip(sample.mutation, sample.context, sample.locus, weighted_phi.T):
            self.mutation_sstats[:, _context, _mutation]+=ss
            self.context_sstats[:, _context]+=ss
            self.locus_sstats[_locus]+=ss

        self.gamma = gamma
        self.weighed_phi = weighted_phi


class CorpusSstats:

    def __init__(self, model_state):

        self.mutation_sstats =  np.zeros_like(model_state.omega)
        self.context_sstats = np.zeros_like(model_state.delta)
        self.locus_sstats = defaultdict(lambda : np.zeros(model_state.n_components))
        self._alpha_sstats = []
        self.exposures = []

    
    def _convert_beta_sstats_to_array(self, k, n_bins):
        arr = np.zeros(n_bins)
        for l,v in self.locus_sstats.items():
            arr[l] = v[k]
        return arr
        
        
    def __add__(self, sstats):

        self.mutation_sstats += sstats.mutation_sstats
        self.context_sstats += sstats.context_sstats
        self._alpha_sstats.append(sstats.gamma)

        for locus, stat in sstats.locus_sstats.items():
            self.locus_sstats[locus] += stat

        #if not self.corpus.shared_exposures:
        #    self._beta_sstats.append(sstats.locus_sstats)
        #    self._exposures.append(sstats.exposures)

        return self
    

    def beta_sstats(self, k, n_bins):
        return self._convert_beta_sstats_to_array(k, n_bins)
    
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
    * For mutation rate estimation need to collect beta_sstats, exposures, and X_matrices

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
        
        '''self.corpus_names = list(corpus_sstats.keys())
        self.beta_sstats = [betas for stats in corpus_sstats.values() for betas in stats.beta_sstats]
        
        self.alpha_sstats = {
            corpus_name : np.array([gamma for gamma in stats.alpha_sstats])
            for corpus_name, stats in corpus_sstats.items()
        }
        
        # is shape I x K x C
        self.context_sstats = np.array([stats.context_sstats for stats in corpus_sstats.values()]) #np.zeros_like(first_corpus.context_sstats)'''
        
        
        #self.context_sstats += corpus.context_sstats
            
