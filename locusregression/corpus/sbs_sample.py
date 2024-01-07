from .featurization import MUTATIONS_IDX, CONTEXT_IDX
from .genome_tools import Region

import os
import subprocess
import tempfile
import sys
from pyfaidx import Fasta
from dataclasses import dataclass
import numpy as np


class SbsRegion(Region):

    @property
    def ref(self):
        return self.annotation[0]

    @property
    def alt(self):
        return self.annotation[1]


complement = {'A' : 'T','T' : 'A','G' : 'C','C' : 'G'}

def revcomp(seq):
    return ''.join(reversed([complement[nuc] for nuc in seq]))

def convert_to_mutation(context, alt):
    
    if not context[1] in 'CT': 
        context, alt = revcomp(context), complement[alt]

    return context, alt


def code_SBS_mutation(*,query_file, fasta_file,
                      chr_prefix = '', output = sys.stdout):

    def _get_context_mutation_idx(fasta_object,*,
                                  chromosome, pos, ref, alt):

        start = pos - 1; end = pos + 2

        try:
            context = fasta_object[chromosome][start : end].seq.upper()
        except KeyError as err:
            raise KeyError('\tChromosome {} found in VCF file is not in the FASTA reference file'\
                .format(chromosome)) from err

        oldnuc = context[1]

        assert oldnuc == ref, '\tLooks like the vcf file was constructed with a different reference genome, different ref allele found at {}:{}, found {} instead of {}'.format(
            chromosome, str(pos), oldnuc, ref 
        )

        context, alt = convert_to_mutation(context, alt)

        return MUTATIONS_IDX[context][alt], CONTEXT_IDX[context], '{c1}[{c2}>{alt}]{c3}'.format(c1 = context[0], c2 = context[1], alt = alt, c3 = context[2])
    

    class QUERY(object):
        CHROM = 0	
        POS = 1
        REF = 2
        ALT = 3
        WEIGHT = 4


    with Fasta(fasta_file) as fa:
    
        for line in query_file:

            if line.startswith('#'):
                continue
            
            line = line.strip().split('\t')

            feature_attr = {
                'chromosome' : chr_prefix + line[QUERY.CHROM],
                'pos' : int(line[QUERY.POS]),
                'ref' : line[QUERY.REF],
                'alt' : line[QUERY.ALT],
            }
            
            mut, context, cosmic_str = _get_context_mutation_idx(
                                            fa, **feature_attr
                                        )
            
            print(
                    feature_attr['chromosome'],
                    feature_attr['pos'],
                    int(feature_attr['pos']) + 1,
                    f'{cosmic_str}:{mut}:{context}:{line[QUERY.WEIGHT]}',
                sep = '\t',
                file = output,
            )


def _process_mapped_sbs_codes(input):
    data = {
        'chrom': [],
        'pos': [],
        'cosmic_str': [],
        'mutation': [],
        'context': [],
        'locus': [],
        'weight' : [],
    }

    for locus_idx, line in enumerate(input):

        chrom, pos, _, _, mutation_codes = line.strip().split('\t')
        if mutation_codes == '.':
            continue

        mutation_codes = [code.split(':') for code in mutation_codes.split(',')]

        for code in mutation_codes:
            data['chrom'].append(chrom)
            data['pos'].append(int(pos))
            data['cosmic_str'].append(code[0])
            data['mutation'].append(int(code[1]))
            data['context'].append(int(code[2]))
            data['weight'].append(float(code[3]))
            data['locus'].append(locus_idx)

    data = {key: np.array(value).astype('S') if key in ['chrom','cosmic_str'] else np.array(value)
            for key, value in data.items()
            }  # Convert lists to numpy arrays

    return data


@dataclass
class SBSSample:

    mutation : np.ndarray
    context : np.ndarray
    locus : np.ndarray
    exposures : np.ndarray
    name : str
    weight : np.ndarray = None
    chrom : np.ndarray = None
    pos : np.ndarray = None
    cosmic_str : np.ndarray = None

    data_attrs = ['chrom','pos','cosmic_str','mutation','context','locus','exposures','weight']
    required = ['mutation','context','locus','exposures']

    @classmethod
    def featurize_mutations(cls, vcf_file, regions_file, fasta_file, exposures,
                        chr_prefix = '', weight_col = None):


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
                    ['bcftools','query','-f','%CHROM\t%POS0\t%REF\t%ALT{0}' + ('\t1\n' if weight_col is None else f'\t%INFO/{weight_col}\n')],
                    stdin = filter_process.stdout,
                    stdout = subprocess.PIPE,
                    stderr = nullout,
                    universal_newlines=True,
                    bufsize=10000,
                )

            '''
            Code mutation converts a line of a vcf file into a mutational code.
            For SBS mutations, the code is a context-mutation-weight tuple, so a line of the ouput would look like:
            
            chr1    1000    1001    12:3:1

            Where 12 is the context code and 3 is the mutation code.
            '''
            code_process = subprocess.Popen(
                ['locusregression','code-sbs',
                    '-fa',fasta_file,
                    '--chr-prefix',chr_prefix
                ],
                stdin=query_process.stdout,
                stdout = subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
            )

            map_process = subprocess.Popen(
                ['bedtools','map',
                '-a', regions_file, 
                '-b', '-', 
                '-sorted',
                '-c','4','-o','collapse', 
                '-delim',','],
                stdin=code_process.stdout,
                stdout=tmp,
                universal_newlines=True,
                bufsize=10000,
            )
            
            map_process.communicate(); code_process.communicate(); filter_process.communicate()
            tmp.flush()
            
            if not all([p.returncode == 0 for p in [map_process, code_process, filter_process]]):
                raise RuntimeError('Sample processing failed. See error message above.')

            with open(tmp.name, 'r') as f:
                return cls(
                    **_process_mapped_sbs_codes(f),
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
        
        mutation_rate = np.ravel( np.zeros_like(self.exposures, dtype = float) )
        
        for locus, weight in zip(self.locus, self.weight):
            mutation_rate[locus] += weight if use_weight else 1.

        mutation_rate = mutation_rate/np.ravel( self.exposures )

        return mutation_rate

