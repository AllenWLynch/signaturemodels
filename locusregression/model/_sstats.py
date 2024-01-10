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

        self._mutation_sstats =  np.zeros_like(model_state.omega)
        self._context_sstats = np.zeros_like(model_state.delta)
        self._locus_sstats = defaultdict(lambda : np.zeros(model_state.n_components))
        self._alpha_sstats = []
        self._beta_sstats = []
        self._exposures = []

        
    def __add__(self, sstats):

        self._mutation_sstats += sstats.mutation_sstats
        self._context_sstats += sstats.context_sstats
        self._alpha_sstats.append(sstats.gamma)

        for locus, stat in sstats.locus_sstats.items():
            self._locus_sstats[locus] += stat

        #if not self.corpus.shared_exposures:
        #    self._beta_sstats.append(sstats.locus_sstats)
        #    self._exposures.append(sstats.exposures)

        return self


    @property
    def mutation_sstats(self):
        return self._mutation_sstats
    
    @property
    def context_sstats(self):
        return self._context_sstats
    
    @property
    def alpha_sstats(self):
        return self._alpha_sstats
    
    @property
    def beta_sstats(self):
        return [self._locus_sstats] #if self.corpus.shared_exposures else self._beta_sstats
    
    @property
    def locus_sstats(self):
        return self._locus_sstats
    


class MetaSstats:
    '''
    De-nest and aggregate sstats across multiple corpuses. 
    * Mutation sstats (omega), context sstats (lambda) can be aggregated across all
    * For mutation rate estimation need to collect beta_sstats, exposures, and X_matrices

    Use the same interface as the CorpusSstats

    '''

    def __init__(self, corpus_sstats):
        
        self.corpus_names = list(corpus_sstats.keys())
        self.beta_sstats = [betas for stats in corpus_sstats.values() for betas in stats.beta_sstats]
        
        self.alpha_sstats = {
            corpus_name : np.array([gamma for gamma in stats.alpha_sstats])
            for corpus_name, stats in corpus_sstats.items()
        }
        
        first_corpus = list(corpus_sstats.values())[0]
        self.mutation_sstats =  np.zeros_like(first_corpus._mutation_sstats)
        self.context_sstats = np.zeros_like(first_corpus._context_sstats)

        for corpus in corpus_sstats.values():
            self.mutation_sstats += corpus.mutation_sstats
            self.context_sstats += corpus.context_sstats
            
