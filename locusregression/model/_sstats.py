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
        self.alpha_sstats = []
        self.exposures = []
        
    def __add__(self, sstats):

        self.mutation_sstats += sstats.mutation_sstats
        self.context_sstats += sstats.context_sstats
        self.alpha_sstats.append(sstats.gamma)

        for locus, stat in sstats.locus_sstats.items():
            self.locus_sstats[locus] += stat

        #if not self.corpus.shared_exposures:
        #    self._beta_sstats.append(sstats.locus_sstats)
        #    self._exposures.append(sstats.exposures)

        return self
    
    @property
    def beta_sstats(self):
        return [self.locus_sstats] #if self.corpus.shared_exposures else self._beta_sstats
    


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
        self.mutation_sstats =  np.zeros_like(first_corpus.mutation_sstats)
        self.context_sstats = np.zeros_like(first_corpus.context_sstats)

        for corpus in corpus_sstats.values():
            self.mutation_sstats += corpus.mutation_sstats
            self.context_sstats += corpus.context_sstats
            
