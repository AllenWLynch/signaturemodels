from .featurization import MUTATIONS_IDX, CONTEXT_IDX
import os
import subprocess
import tempfile
import sys
from pyfaidx import Fasta
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix

complement = {'A' : 'T','T' : 'A','G' : 'C','C' : 'G'}

def revcomp(seq):
    return ''.join(reversed([complement[nuc] for nuc in seq]))


def convert_to_mutation(context, alt):
    
    if not context[1] in 'CT': 
        context, alt = revcomp(context), complement[alt]

    return context, alt


@dataclass
class SBSSample:

    mutation : np.ndarray
    context : np.ndarray
    locus : np.ndarray
    exposures : np.ndarray
    name : str
    chrom : np.ndarray = None
    pos : np.ndarray = None
    weight : np.ndarray = None
    
    type_map = {
        'mutation' : np.uint8,
        'context' : np.uint8,
        'locus' : np.uint32,
        'exposures' : np.float32,
        'chrom' : 'S',
        'pos' : np.uint32,
        'weight' : np.float32,
        'name' : 'S',
    }

    data_attrs = ['mutation','context','locus','exposures','weight','chrom','pos']
    required = ['mutation','context','locus','exposures']

         
    @classmethod
    def featurize_mutations(cls, vcf_file, regions_file, fasta_file, exposures,
                        chr_prefix = '', weight_col = None):
        

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

            assert oldnuc == ref, '\tLooks like the vcf file was constructed with a different reference genome, different ref allele found at {}:{}, found {} instead of {}'.format(
                chrom, str(pos), oldnuc, ref 
            )

            context, alt = convert_to_mutation(context, alt)

            {
                'chrom' : chrom,
                'locus' : locus_idx,
                'mutation' : MUTATIONS_IDX[context][alt],
                'context' : CONTEXT_IDX[context],
                'weight' : float(weight),
                'pos' : int(pos),
            }



        with tempfile.NamedTemporaryFile() as tmp:

            filter_process = subprocess.Popen(
                ['bcftools','view','-v','snps', vcf_file],
                stdout = subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
                stderr = subprocess.DEVNULL,
            )
            
            with open(os.devnull, "w") as nullout:
                query_process = subprocess.Popen(
                    ['bcftools','query','-f','%CHROM\t%POS0\t%POS0\t%CHROM|%POS0|%REF|%ALT{0}|' \
                        + ('1' if weight_col is None else f'%INFO/{weight_col}') + '\n'],
                    stdin = filter_process.stdout,
                    stdout = subprocess.PIPE,
                    stderr = nullout,
                    universal_newlines=True,
                    bufsize=10000,
                )


            subprocess.check_output(
                ['bedtools',
                 'intersect',
                '-a', regions_file, 
                '-b', '-', 
                '-sorted',
                '-wa','-wb',
                '-split'],
                stdin=query_process.stdout,
                stdout=tmp,
                universal_newlines=True,
                bufsize=10000,
            )
            
            query_process.communicate(); filter_process.communicate()
            tmp.flush()
            
            #if not all([p.returncode == 0 for p in [query_process, filter_process]]):
            #    raise RuntimeError('Sample processing failed. See error message above.')

            mutations = defaultdict(list)

            with open(tmp.name, 'r') as f, \
                Fasta(fasta_file) as fa:
                
                for line in f:
                    for k, v in process_line(line, fa):
                        mutations[k].append(v)

            


            return cls(
                **cls.process_mutations(f),
                name = os.path.abspath(vcf_file),
                exposures = np.array(exposures),
            )
            
    def __len__(self):
        return len(self.locus)
    
    
    @property
    def n_mutations(self):
        return sum(self.weight)
            
            
    def create_h5_dataset(self, h5_object, dataset_name):

        for attr in self.data_attrs:
            h5_object.create_dataset(f'{dataset_name}/{attr}', data = self.__getattribute__(attr))

        h5_object[dataset_name].attrs['name'] = self.name

    
    @classmethod
    def read_h5_dataset(cls, h5_object, dataset_name, read_optional = True):

        read_attrs = cls.data_attrs if read_optional else cls.required

        return SBSSample(
            **{
                attr : h5_object[f'{dataset_name}/{attr}'][...]
                for attr in read_attrs
            },
            name = h5_object[dataset_name].attrs['name'],
        )
    

    def asdict(self):
        return {
            **{attr : self.__getattribute__(attr) for attr in self.data_attrs},
            'name' : self.name,
        }
        
    
    def get_empirical_mutation_rate(self, use_weight = True):

        n_loci = self.exposures.shape[1]
        
        mutations = coo_matrix(
            (self.weight, (self.context, self.locus)),
            shape = (32, n_loci),
            dtype = np.uint8
        )
        
        mutations = mutations.multiply(1/self.exposures)

        return mutations.todok()

