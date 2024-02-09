from itertools import product
from ..sample import Sample
import numpy as np
import matplotlib.pyplot as plt

complement = {'A' : 'T','T' : 'A','G' : 'C','C' : 'G'}

def revcomp(seq):
    return ''.join(reversed([complement[nuc] for nuc in seq]))


def convert_to_cosmic(context, alt):

    if not context[1] in 'CT': 
        context, alt = revcomp(context), complement[alt]

    return context, alt


CONTEXTS = sorted(
    map(lambda x : ''.join(x), product('ATCG','ATCG','ATCG')), 
    key = lambda x : (x[1], x[0], x[2])
    )


CONTEXT_IDX = dict(zip(CONTEXTS, range(len(CONTEXTS))))

MUTATIONS = {
    context : [alt for alt in 'ACGT' if not alt == context[1]] # nucleotide cannot mutate to itself
    for context in CONTEXTS
}

MUTATIONS_IDX = {
    context : dict(zip(m, range(len(MUTATIONS))))
    for context, m in MUTATIONS.items()
}

COSMIC_SORT_ORDER = [
 'A[C>A]A',
 'A[C>A]C',
 'A[C>A]G',
 'A[C>A]T',
 'C[C>A]A',
 'C[C>A]C',
 'C[C>A]G',
 'C[C>A]T',
 'G[C>A]A',
 'G[C>A]C',
 'G[C>A]G',
 'G[C>A]T',
 'T[C>A]A',
 'T[C>A]C',
 'T[C>A]G',
 'T[C>A]T',
 'A[C>G]A',
 'A[C>G]C',
 'A[C>G]G',
 'A[C>G]T',
 'C[C>G]A',
 'C[C>G]C',
 'C[C>G]G',
 'C[C>G]T',
 'G[C>G]A',
 'G[C>G]C',
 'G[C>G]G',
 'G[C>G]T',
 'T[C>G]A',
 'T[C>G]C',
 'T[C>G]G',
 'T[C>G]T',
 'A[C>T]A',
 'A[C>T]C',
 'A[C>T]G',
 'A[C>T]T',
 'C[C>T]A',
 'C[C>T]C',
 'C[C>T]G',
 'C[C>T]T',
 'G[C>T]A',
 'G[C>T]C',
 'G[C>T]G',
 'G[C>T]T',
 'T[C>T]A',
 'T[C>T]C',
 'T[C>T]G',
 'T[C>T]T',
 'A[T>A]A',
 'A[T>A]C',
 'A[T>A]G',
 'A[T>A]T',
 'C[T>A]A',
 'C[T>A]C',
 'C[T>A]G',
 'C[T>A]T',
 'G[T>A]A',
 'G[T>A]C',
 'G[T>A]G',
 'G[T>A]T',
 'T[T>A]A',
 'T[T>A]C',
 'T[T>A]G',
 'T[T>A]T',
 'A[T>C]A',
 'A[T>C]C',
 'A[T>C]G',
 'A[T>C]T',
 'C[T>C]A',
 'C[T>C]C',
 'C[T>C]G',
 'C[T>C]T',
 'G[T>C]A',
 'G[T>C]C',
 'G[T>C]G',
 'G[T>C]T',
 'T[T>C]A',
 'T[T>C]C',
 'T[T>C]G',
 'T[T>C]T',
 'A[T>G]A',
 'A[T>G]C',
 'A[T>G]G',
 'A[T>G]T',
 'C[T>G]A',
 'C[T>G]C',
 'C[T>G]G',
 'C[T>G]T',
 'G[T>G]A',
 'G[T>G]C',
 'G[T>G]G',
 'G[T>G]T',
 'T[T>G]A',
 'T[T>G]C',
 'T[T>G]G',
 'T[T>G]T']

_transition_palette = {
    ('C','A') : (0.33, 0.75, 0.98),
    ('C','G') : (0.0, 0.0, 0.0),
    ('C','T') : (0.85, 0.25, 0.22),
    ('T','A') : (0.78, 0.78, 0.78),
    ('T','C') : (0.51, 0.79, 0.24),
    ('T','G') : (0.89, 0.67, 0.72)
}

MUTATION_PALETTE = [color for color in _transition_palette.values() for i in range(16)]


class SBSSample(Sample):

    N_CONTEXTS=64
    N_MUTATIONS=3
    N_ATTRIBUTES=1


    def plot(self, ax=None, figsize=(5,3), show_strand=True,**kwargs):
        
        context_dist = np.zeros((self.N_CONTEXTS,))
        mutation_dist = np.zeros((self.N_CONTEXTS, self.N_MUTATIONS,))

        for context_idx, mutation_idx, weight in zip(
            self.context, self.mutation, self.weight
        ):
            context_dist[context_idx] += weight
            mutation_dist[context_idx, mutation_idx] += weight

        return self.plot_factorized(context_dist, mutation_dist, None, 
                                   ax=ax, 
                                   figsize=figsize,
                                   show_strand=show_strand, 
                                   **kwargs)
        

    @staticmethod
    def plot_factorized(context_dist, mutation_dist, attribute_dist, 
                        ax=None, figsize=(5,3), show_strand=True,**kwargs):

        def to_cosmic_str(context, mutation):
            return f'{context[0]}[{context[1]}>{mutation}]{context[2]}'

        joint_prob = (context_dist[:,None]*mutation_dist).ravel() # CxM
        event_name = [(to_cosmic_str(c,m),'f') if c[1] in 'TC' else (to_cosmic_str(revcomp(c), complement[m]), 'r')
                      for c in CONTEXTS for m in MUTATIONS[c]
                     ]
        
        event_prob = dict(zip(event_name, joint_prob))

        fwd_events = np.array([event_prob[(event, 'f')] for event in COSMIC_SORT_ORDER])
        rev_events = np.array([event_prob[(event, 'r')] for event in COSMIC_SORT_ORDER])

        
        if ax is None:
            _, ax = plt.subplots(1,1,figsize= figsize)

        plot_kw = dict(
            x = COSMIC_SORT_ORDER,
            color = MUTATION_PALETTE,
            width = 1,
            edgecolor = 'white',
            linewidth = 0.5,
            error_kw = {'alpha' : 0.5, 'linewidth' : 0.5}
        )
        extent = max(joint_prob)

        if show_strand:
            ax.bar(height = fwd_events, **plot_kw)
            ax.bar(height = -rev_events, **plot_kw)
            
            ax.set(yticks = [-extent,0,extent], xticks = [], 
               xlim = (-1,96), ylim = (-1.1*extent, 1.1*extent)
            )  
        else:
            ax.bar(height = fwd_events + rev_events, **plot_kw)
            ax.set(yticks = [0,extent], xticks = [], 
               xlim = (-1,96), ylim = (0, 1.1*extent)
            )

        ax.axhline(0, color = 'lightgrey', linewidth = 0.25)

        for s in ['left','right','top','bottom']:
            ax.spines[s].set_visible(False)

        return ax