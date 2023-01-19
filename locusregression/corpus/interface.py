
from functools import wraps


def get_corpus_lists(func):

    @wraps(func)
    def extract_data(model, corpus):
        return func(model, **{
            'X_matrix' : corpus.genome_features.T,
            'mutation' : corpus.mutations,
            'context' : corpus.contexts,
            'locus' : corpus.loci,
            'count' : corpus.counts,
            'trinuc_distributions' : corpus._trinuc_distributions.T,
            'window_size' : corpus.window_sizes
        })

    return extract_data



