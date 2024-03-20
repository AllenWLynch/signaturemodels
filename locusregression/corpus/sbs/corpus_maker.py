
from ..corpus_maker import CorpusMaker
from pyfaidx import Fasta
import numpy as np
from collections import Counter, defaultdict
import logging
import tqdm
from .observation_config import SBSSample, MUTATIONS_IDX, CONTEXT_IDX, CONTEXTS, \
            convert_to_cosmic, revcomp
import subprocess
import os
from joblib import Parallel, delayed
from functools import partial
import tempfile
import sys
logger = logging.getLogger('DataReader')
logger.setLevel(logging.INFO)


def get_passed_SNVs(vcf_file, query_string, 
                    output=subprocess.PIPE,
                    filter_string=None,
                    sample=None,):
    
    filter_basecmd = [
        'bcftools','view',
        '-f','PASS',
        '-v','snps'
    ]
    
    if not sample is None:
        filter_basecmd += ['-s', sample]

    if not filter_string is None:
        filter_basecmd += ['-i', filter_string]

    filter_process = subprocess.Popen(
                filter_basecmd + [vcf_file],
                stdout = subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
                stderr = sys.stderr,
            )
        
    query_process = subprocess.Popen(
        ['bcftools','query','-f', query_string],
        stdin = filter_process.stdout,
        stdout = output,
        stderr = sys.stderr,
        universal_newlines=True,
        bufsize=10000,
    )

    return query_process


class WeirdMutationError(Exception):
    pass


class SBSCorpusMaker(CorpusMaker):

    corpus_type = 'SBS'

    
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
        

            return SBSSample(
                **mutations,
                name = os.path.abspath(vcf_file),
            )
        
        finally:
            if not mutation_rate_file is None:
                clustered_vcf.close()


    @classmethod
    def collect_samples(cls,sample_files, 
                     weight_col = None, 
                     chr_prefix = '',
                     n_jobs=1,*,
                     fasta_file, regions_file, exposures):
        
        logger.info('Reading VCF files ...')
        samples = []

        if exposures.shape[0] == 1:
            
            def duplication_iterator(i):
                for _ in range(i):
                    yield exposures

            _exposures = duplication_iterator(len(sample_files))

        else:
            _exposures = list(exposures)

            assert len(_exposures) == len(sample_files), \
                'If providing exposures, provide one for the whole corpus, or one per sample.'


        with tempfile.NamedTemporaryFile() as mutrate_file, \
            tempfile.NamedTemporaryFile() as genome_file:

            subprocess.check_call(
                ['faidx', fasta_file, '-i', 'chromsizes', '-o', genome_file.name]
            )
            
            subprocess.check_call(
                ['locusregression','preprocess-estimate-mutrate',
                '--vcf-files', *sample_files,
                '--chr-prefix', chr_prefix,
                '--genome-file',genome_file.name,
                '-o', mutrate_file.name,
                ]
            )

            read_vcf_fn = partial(
                cls.featurize_mutations,
                regions_file = regions_file, 
                fasta_file = fasta_file,
                chr_prefix = chr_prefix, 
                weight_col = weight_col,
                mutation_rate_file = mutrate_file.name,
            )

            samples = Parallel(
                            n_jobs = n_jobs, 
                            verbose = 10,
                        )(
                            delayed(read_vcf_fn)(vcf, exposures = sample_exposure)
                            for vcf, sample_exposure in zip(sample_files, _exposures)
                        )
        
        logger.info('Done reading VCF files.')
        return samples



    @staticmethod
    def get_context_frequencies(window_set, fasta_file, n_jobs = 1):
        
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

