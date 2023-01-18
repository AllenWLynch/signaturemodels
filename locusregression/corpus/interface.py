
from functools import wraps


def get_corpus_lists(func):

    @wraps(func)
    def extract_data(model, corpus):
        return {
            'X_matrix' : corpus.genome_features,
            'mutation' : corpus.mutation,
            'context' : corpus.contexts,
            'locus' : corpus.loci,
            'count' : corpus.counts,
            'trinuc_distributions' : corpus._trinuc_distributions,
            'window_size' : corpus.window_sizes
        }

    return extract_data



