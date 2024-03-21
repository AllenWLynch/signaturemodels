from itertools import product
from ..sample import Sample
import numpy as np
import matplotlib.pyplot as plt
from .mutation_preprocessing import get_passed_SNVs
from pyfaidx import Fasta
import numpy as np
from collections import Counter, defaultdict
import logging
import tqdm
import subprocess
import os
import tempfile
logger = logging.getLogger('SBS-DataReader')
logger.setLevel(logging.INFO)

complement = {'A' : 'T','T' : 'A','G' : 'C','C' : 'G'}

def revcomp(seq):
    return ''.join(reversed([complement[nuc] for nuc in seq]))


def convert_to_cosmic(context, alt):

    cardinality = 0
    if not context[1] in 'CT': 
        cardinality = 1
        context, alt = revcomp(context), complement[alt]

    return context, alt, cardinality


nucleotide_order = ['C','T','A','G']

CONTEXTS = sorted(
    map(lambda x : ''.join(x), product('ATCG','TC','ATCG')), 
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

class WeirdMutationError(Exception):
    pass


class SBSSample(Sample):

    N_CARDINALITY=2
    N_CONTEXTS=32
    N_MUTATIONS=3
    N_ATTRIBUTES=1


    def plot(self, ax=None, figsize=(5.5,1.25), show_strand=False,
             normalizer = None, **kwargs):
        
        context_dist = np.zeros((self.N_CONTEXTS,))
        mutation_dist = np.zeros((self.N_CONTEXTS, self.N_MUTATIONS,))

        for context_idx, mutation_idx, weight in zip(
            self.context, self.mutation, self.weight
        ):
            context_dist[context_idx] += weight
            mutation_dist[context_idx, mutation_idx] += weight

        if not normalizer is None:
            context_dist = context_dist/normalizer

        return self.plot_factorized(context_dist, mutation_dist, None, 
                                   ax=ax, 
                                   figsize=figsize,
                                   show_strand=show_strand, 
                                   **kwargs)
        

    @staticmethod
    def plot_factorized(context_dist, mutation_dist, attribute_dist, 
                        ax=None, figsize=(5.5,1.25), show_strand=False,**kwargs):

        def to_cosmic_str(context, mutation):
            return f'{context[0]}[{context[1]}>{mutation}]{context[2]}'

        joint_prob = (context_dist[:,None]*mutation_dist).ravel() # CxM
        event_name = [(to_cosmic_str(c,m),'f') if c[1] in 'TC' else (to_cosmic_str(revcomp(c), complement[m]), 'r')
                      for c in CONTEXTS for m in MUTATIONS[c]
                     ]
        
        event_prob = dict(zip(event_name, joint_prob))

        fwd_events = np.array([event_prob[(event, 'f')] for event in COSMIC_SORT_ORDER])
        #rev_events = np.array([event_prob[(event, 'r')] for event in COSMIC_SORT_ORDER])

        
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

        ax.bar(height = fwd_events, **plot_kw)
        ax.set(yticks = [], xticks = [], 
            xlim = (-1,96), ylim = (0, 1.1*extent)
        )

        ax.axhline(0, color = 'lightgrey', linewidth = 0.25)

        for s in ['left','right','top','bottom']:
            ax.spines[s].set_visible(False)

        return ax
    

    @classmethod
    def featurize_mutations(cls, 
                    vcf_file, regions_file, fasta_file,
                    chr_prefix = '', 
                    weight_col = None, 
                    mutation_rate_file=None
                ):


        def process_line(line, fasta_object):

            fields = line.strip().split('\t')
            chrom=fields[0]; locus_idx=int(fields[3]); mutation_code=fields[-1]

            pos, ref, alt, weight = mutation_code.split('|')
            pos = int(pos)

            start = pos - 1; end = pos + 2
            try:
                context = fasta_object[chrom][start : end].seq.upper()
            except KeyError as err:
                raise KeyError('\tChromosome {} found in VCF file is not in the FASTA reference file'\
                    .format(chrom)) from err

            oldnuc = context[1]

            if not ref.upper() in 'ATCG' or not alt.upper() in 'ATCG':
                raise WeirdMutationError('\tWeird mutation found at {}:{} {}->{}'.format(
                    chrom, str(pos), ref, alt
                ))

            assert oldnuc == ref, '\tLooks like the vcf file was constructed with a different reference genome, different ref allele found at {}:{}, found {} instead of {}'.format(
                chrom, str(pos), oldnuc, ref 
            )

            context, alt, cardinality = convert_to_cosmic(context, alt)

            return {
                'chrom' : chrom,
                'locus' : locus_idx,
                'mutation' : MUTATIONS_IDX[context][alt],
                'context' : CONTEXT_IDX[context],
                'cardinality' : int(cardinality),
                'attribute' : 0, # placeholder for now
                'weight' : float(weight),
                'pos' : int(pos),
            }


        query_statement = chr_prefix + '%CHROM\t%POS0\t%POS\t%POS0|%REF|%ALT|' \
                                                    + ('1\n' if weight_col is None else f'%INFO/{weight_col}\n')
        
        try:
            
            if not mutation_rate_file is None:

                clustered_vcf = tempfile.NamedTemporaryFile()
                
                with open(clustered_vcf.name, 'w') as f:
                    subprocess.check_call(
                        ['locusregression','preprocess-cluster-mutations',
                        vcf_file,
                        '-m', mutation_rate_file,
                        '--chr-prefix', chr_prefix],
                        stdout = f,
                    )

                query_process = get_passed_SNVs(clustered_vcf.name, query_statement,
                                                filter_string='clusterSize<3',
                                                )

                clustered_mutations_vcf, _ = get_passed_SNVs(clustered_vcf.name, query_statement,
                                                filter_string='clusterSize>=3',
                                                ).communicate()
                n_cluster_mutations = len(clustered_mutations_vcf.split("\n"))

            else:
                query_process = get_passed_SNVs(vcf_file, query_statement)
                n_cluster_mutations = 0

            intersect_process = subprocess.Popen(
                ['bedtools',
                'intersect',
                '-a', regions_file, 
                '-b', '-', 
                '-sorted',
                '-wa','-wb',
                '-split'],
                stdin=query_process.stdout,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
            )

            mutations = defaultdict(list)
            with Fasta(fasta_file) as fa:

                while True:
                    line = intersect_process.stdout.readline()

                    if not line:
                        break
                    
                    try:
                        for k, v in process_line(line, fa).items():
                            mutations[k].append(v)
                    except WeirdMutationError as err:
                        logger.warning(err)
                        continue

            intersect_process.communicate()

            for k, v in mutations.items():
                mutations[k] = np.array(v).astype(SBSSample.type_map[k])

            if n_cluster_mutations > 0:
                total_mutations = n_cluster_mutations + len(mutations[k])
                logger.info(
                    f'Filtered {n_cluster_mutations} cluster mutations (out of {total_mutations}).'
                )
        

            return cls(
                **mutations,
                name = os.path.abspath(vcf_file),
            )
        
        finally:
            if not mutation_rate_file is None:
                clustered_vcf.close()


    @classmethod
    def get_context_frequencies(cls, window_set, fasta_file, n_jobs = 1):
    
        def count_trinucs(bed12_region, fasta_object):

            def rolling(seq, w = 3):
                for i in range(len(seq) - w + 1):
                    yield seq[ i : i+w]

            trinuc_counts = Counter()
            N_counts=0

            for chrom, start, end in bed12_region.segments():

                window_sequence = fasta_object[chrom][max(start-1,0) : end+1].seq.upper()
                
                for trinuc in rolling(window_sequence):
                    if not 'N' in trinuc:
                        trinuc_counts[trinuc]+=1
                    else:
                        N_counts+=1

            pseudocount = N_counts/(2*32)

            return [
                [trinuc_counts[context]+pseudocount for context in CONTEXTS],
                [trinuc_counts[revcomp(context)]+pseudocount for context in CONTEXTS]
            ]
        
        with Fasta(fasta_file) as fasta_object:
            trinuc_matrix = [
                count_trinucs(w, fasta_object) 
                for w in tqdm.tqdm(window_set, nrows=100, desc = 'Aggregating trinucleotide content')
            ]

        # L, D, C
        trinuc_matrix = np.array(trinuc_matrix).transpose(((1,2,0))) # DON'T (!) add a pseudocount

        return trinuc_matrix # DON'T (!) normalize, the number of contexts in a window is part of the likelihood
    
