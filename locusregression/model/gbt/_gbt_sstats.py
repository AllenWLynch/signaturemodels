import numpy as np
from collections import defaultdict
import locusregression.model._sstats as base_sstats

def repeat_iterator(X, n):
    class reuseiterator:
        def __iter__(self):
            for i in range(n):
                yield X

    return reuseiterator()

class SampleSstats(base_sstats.SampleSstats):
    pass

class CorpusSstats(base_sstats.CorpusSstats):

    def __init__(self, corpus, model_state, corpus_state):

        self._mutation_sstats =  np.zeros_like(model_state.omega)
        self._context_sstats = np.zeros_like(model_state.delta)
        self._locus_sstats = defaultdict(lambda : np.zeros(model_state.n_components))
        self._alpha_sstats = []

        self.corpus = corpus
        self._logmu = corpus_state._logmu.copy()
        self._lognu = corpus_state._logmu.copy()

        if self.corpus.shared_exposures:
            self._exposures = [self.corpus.exposures]
        else:
            self._beta_sstats = []
            self._exposures = []
    
    @property
    def logmus(self):
        return self._logmu
    
    @property
    def lognus(self):
        return self._lognu


class MetaSstats(base_sstats.MetaSstats):
    '''
    De-nest and aggregate sstats across multiple corpuses. 
    * Mutation sstats (omega), context sstats (lambda) can be aggregated across all
    * For mutation rate estimation need to collect beta_sstats, exposures, and X_matrices

    Use the same interface as the CorpusSstats

    '''

    def __init__(self, corpus_sstats, model_state):
        
        self.beta_sstats = [betas for stats in corpus_sstats.values() for betas in stats.beta_sstats]
        self.exposures = [exp for stats in corpus_sstats.values() for exp in stats.exposures]
        self.X_matrices = [X for stats in corpus_sstats.values() for X in stats.X_matrices]
        self.logmus = [sstats.logmus for sstats in corpus_sstats.values()]
        self.lognus = [sstats.lognus for sstats in corpus_sstats.values()]
        

        self.alpha_sstats = {corpus_name : np.array([gamma for gamma in stats.alpha_sstats])
                            for corpus_name, stats in corpus_sstats.items()}
        
        first_corpus = list(corpus_sstats.values())[0]
        self.trinuc_distributions = first_corpus.corpus.trinuc_distributions

        self.mutation_sstats =  np.zeros_like(first_corpus._mutation_sstats)
        self.context_sstats = np.zeros_like(first_corpus._context_sstats)
        self.locus_sstats = defaultdict(lambda : np.zeros(model_state.n_components))

        for corpus in corpus_sstats.values():
            self.mutation_sstats += corpus.mutation_sstats
            self.context_sstats += corpus.context_sstats
            
            for locus, stat in corpus.locus_sstats.items():
                self.locus_sstats[locus] += stat
