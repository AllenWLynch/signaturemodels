from .featurization import MUTATIONS_IDX, CONTEXT_IDX
from .genome_tools import Region

import os
import subprocess
import tempfile
import sys
from pyfaidx import Fasta
from dataclasses import dataclass
import numpy as np


class VCF(object):
    CHROM = 0	
    POS = 1
    REF = 3
    ALT = 4

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


def code_SBS_mutation(*,vcf_file, fasta_file, index = -1, 
                      chr_prefix = '', sep = '\t', output = sys.stdout):

    def _get_context_mutation_idx(fasta_object, sbs):

        try:
            context = fasta_object[sbs.chromosome][sbs.start : sbs.end].seq.upper()
        except KeyError as err:
            raise KeyError('\tChromosome {} found in VCF file is not in the FASTA reference file'\
                .format(sbs.chromosome)) from err

        newref = context[1]

        assert newref == sbs.ref, '\tLooks like the vcf file was constructed with a different reference genome, different ref allele found at {}:{}, found {} instead of {}'.format(
            sbs.chromosome, str(sbs.start+1), newref, sbs.ref 
        )

        context, alt = convert_to_mutation(context, sbs.alt)

        return MUTATIONS_IDX[context][alt], CONTEXT_IDX[context], '{c1}[{c2}>{alt}]{c3}'.format(c1 = context[0], c2 = context[1], alt = alt, c3 = context[2])
    

    with Fasta(fasta_file) as fa:
    
        for line in vcf_file:
            
            if line.startswith('#'):
                continue
            
            line = line.strip().split(sep)
            mut_region = SbsRegion(
                                    chr_prefix + line[VCF.CHROM],int(line[VCF.POS])-1+index, int(line[VCF.POS])+2+index, # add some buffer indices for the context
                                    annotation=(line[VCF.REF], line[VCF.ALT])
                                )
            
            mut, context, cosmic_str = _get_context_mutation_idx(fa, mut_region)

            print(
                mut_region.chromosome, mut_region.start, mut_region.end,f'{mut_region.chromosome}:{mut_region.start}:{cosmic_str}:{mut}:{context}',
                sep = '\t',
                file = output,
            )


@dataclass
class SBSSample:

    mutation : np.ndarray
    context : np.ndarray
    locus : np.ndarray
    exposures : np.ndarray
    name : str
    chrom : np.ndarray = None
    pos : np.ndarray = None
    cosmic_str : np.ndarray = None

    data_attrs = ['chrom','pos','cosmic_str','mutation','context','locus','exposures']
    required = ['mutation','context','locus','exposures']


    @classmethod
    def featurize_mutations(cls, vcf_file, regions_file, fasta_file, exposures,
                        sep = '\t', index = -1, chr_prefix = ''):
        
        with tempfile.NamedTemporaryFile() as tmp:

            filter_process = subprocess.Popen(
                ['bcftools','view','-v','snps','-O','v', vcf_file],
                stdout = subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
                stderr = subprocess.DEVNULL,
            )


            '''
            Code mutation converts a line of a vcf file into a mutational code.
            For SBS mutations, the code is a context-mutation pair, so a line of the ouput would look like:
            
            chr1    1000    1001    12:3

            Where 12 is the context code and 3 is the mutation code.
            '''
            code_process = subprocess.Popen(
                ['locusregression','code-sbs',
                    '-fa',fasta_file,
                    '-sep', sep,
                    '--index',str(index),
                    '--chr-prefix',chr_prefix],
                stdin=filter_process.stdout,
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

            chrom=[]; pos=[]; cosmic_str=[]; mutations=[]; contexts=[]; loci=[]

            with open(tmp.name, 'r') as f:
                
                for locus_idx, line in enumerate(f):

                    line = line.strip().split('\t')
                    if line[-1] == '.':
                        continue

                    mutation_codes = line[-1].split(',')
                    mutation_codes = [code.split(':') for code in mutation_codes]

                    for code in mutation_codes:
                        chrom.append(code[0])
                        pos.append(int(code[1]) + 2)
                        cosmic_str.append(code[2])
                        mutations.append(int(code[3]))
                        contexts.append(int(code[4]))
                        loci.append(locus_idx)
                        

        return cls(
            mutation = np.array(mutations), 
            context = np.array(contexts),
            locus = np.array(loci),
            cosmic_str = np.array(cosmic_str).astype('S'),
            chrom = np.array(chrom).astype('S'),
            pos = np.array(pos),
            name = os.path.abspath(vcf_file),
            exposures = np.array(exposures),
        )
    
    
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
        
