
from .genome_tools import Region, RegionSet, Genome
from .featurization import CONTEXT_IDX, MUTATIONS_IDX
from .corpus import Corpus, InMemorySamples
from pyfaidx import Fasta
import numpy as np
from collections import Counter
import logging
import tqdm
import subprocess
import tempfile
import sys
logger = logging.getLogger('DataReader')
logger.setLevel(logging.INFO)


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



class VCF(object):
    CHROM = 0	
    POS = 1
    REF = 3
    ALT = 4


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

        return MUTATIONS_IDX[context][alt], CONTEXT_IDX[context]
    

    with Fasta(fasta_file) as fa:
    
        for line in vcf_file:
            
            if line.startswith('#'):
                continue
            
            line = line.strip().split(sep)
            mut_region = SbsRegion(
                                    chr_prefix + line[VCF.CHROM],int(line[VCF.POS])-1+index, int(line[VCF.POS])+2+index, # add some buffer indices for the context
                                    annotation=(line[VCF.REF], line[VCF.ALT])
                                )
            
            mut, context = _get_context_mutation_idx(fa, mut_region)

            print(
                mut_region.chromosome, mut_region.start, mut_region.end,f'{mut}:{context}',
                sep = '\t',
                file = output,
            )



class Sample:

    @staticmethod
    def featurize_mutations(vcf_file, regions_file, fasta_file,
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

            mutations=[]; contexts=[]; loci=[]; counts=[]

            with open(tmp.name, 'r') as f:
                
                for locus_idx, line in enumerate(f):

                    line = line.strip().split('\t')
                    if line[-1] == '.':
                        continue

                    mutation_codes = line[-1].split(',')
                    mutation_codes = [tuple(map(int, code.split(':'))) for code in mutation_codes]
                    codes_and_counts = Counter(mutation_codes)

                    for code, count in codes_and_counts.items():
                        mutations.append(code[0])
                        contexts.append(code[1])
                        loci.append(locus_idx)
                        counts.append(count)

        return {
            'mutation' : np.array(mutations), 
            'context' : np.array(contexts),
            'locus' : np.array(loci), 
            'count' : np.array(counts)
        }
      


class CorpusReader:

    corpus_class = Corpus

    @classmethod
    def create_trinuc_file(cls,
            fasta_file,
            regions_file,
            output,
            n_jobs = 1,
            sep = '\t',
        ):

        windows = cls.read_windows(regions_file, None, sep = sep)

        with Fasta(fasta_file) as fa:
            trinuc_distributions = cls.get_trinucleotide_distributions(
                        windows, fa, n_jobs=n_jobs
                    )
            
        np.savez(output, x = trinuc_distributions)



    @classmethod
    def create_corpus(cls, 
            sep = '\t', 
            index = -1,
            chr_prefix = '', 
            n_jobs = 1,
            exposure_files = None,
            trinuc_file = None,*,
            correlates_file,
            fasta_file,
            vcf_files,
            regions_file,
            corpus_name,
        ):

        if exposure_files is None:
            exposure_files = []

        elif len(exposure_files) > 1:
            assert len(exposure_files) == len(vcf_files), \
                'If providing exposure files, provide one for the whole corpus, or one per sample.'
            

        windows = cls.read_windows(regions_file, None, sep = sep)

        if len(exposure_files) > 0:
            exposures = cls.calculate_exposures(
                windows, 
                cls.collect_exposures(exposure_files, windows)
            )
        else:
            exposures = cls.calculate_exposures(
                windows, 
                np.ones((1, len(windows)))
            )
        
        features, feature_names = cls.read_correlates(
            correlates_file, windows, required_columns=None
        )

        samples = cls.collect_vcfs(
            vcf_files = vcf_files, 
            fasta_file = fasta_file, 
            regions_file = regions_file,
            index = index,
            sep = sep,
            chr_prefix = chr_prefix,
        )

        trinuc_distributions = None
        if trinuc_file is None:
            with Fasta(fasta_file) as fa:
                trinuc_distributions = cls.get_trinucleotide_distributions(
                            windows, fa, n_jobs=n_jobs
                        )    
        else:
            trinuc_distributions = cls.read_trinuc_distribution(trinuc_file)

            assert trinuc_distributions.shape[0] == len(windows), \
                'The number of trinucleotide distributions provided in {} does not match the number of specified windows.\n'\

        shared = True
        if exposures.shape[0] == 1:
            samples = [{**sample, 'exposures' : exposures} 
                    for sample in samples]
            
        else:
            shared = False
            samples = [
                {**sample, 'exposures' : exposure[None,:]}
                for sample, exposure in zip(samples, exposures)
            ]

        return Corpus(
            samples = InMemorySamples(samples),
            feature_names = feature_names,
            X_matrix = features.T,
            trinuc_distributions = trinuc_distributions.T,
            shared_exposures = shared,
            name = corpus_name,
        )


    @staticmethod
    def read_trinuc_distribution(trinuc_file):
        return np.load(trinuc_file)['x']


    @staticmethod
    def read_exposure_file(exposure_file, windows):
        
        exposures = []
        with open(exposure_file, 'r') as f:

            for lineno, txt in enumerate(f):

                if not txt[0] == '#':

                    try:
                        exposures.append( float(txt.strip()) )
                    except ValueError as err:
                        raise ValueError('Record {} on line {} could not be converted to float dtype.'.format(
                            txt, lineno,
                        )) from err

        assert len(exposures) == len(windows), 'The number of exposures provided in {} does not match the number of specified windows.\n'\
                'Each window must have correlates provided.'

        assert all([e >= 0. for e in exposures]), 'Exposures must be non-negative.'

        return np.array(exposures)
                
        
    @staticmethod
    def read_correlates(correlates_file, windows, 
        required_columns = None, sep = '\t'):
        
        logger.info('Reading genomic features ...')
        
        numcols = None
        correlates = []
        with open(correlates_file, 'r') as f:
            
            for lineno, txt in enumerate(f):
                
                line = txt.strip().split(sep)

                if lineno == 0:
                    numcols = len(line)
                    if not txt[0] == '#':
                        logger.warn(
                            'The first line of the tsv file does not start with colnames prefixed with "#".\n'
                            'e.g. #chr\t#start\t#end\t#feature1 ... etc.'
                            'The first line of the file will be treated as column names.'
                            )

                    assert len(line) >= 1, 'Cannot run algorithm without at least one feature provided.'
                    columns = [col.strip('#') for col in line]

                    if required_columns is None:
                        logger.info('Using correlates: ' + ', '.join(columns))
                    else:
                        missing_cols = set(required_columns).difference(columns)

                        if len(missing_cols) > 0:
                            raise ValueError('The required correlate {} was missing from file: {}.\n All correlates must be specified for all samples'.format(
                                    's ' + ', '.join(missing_cols) if len(missing_cols) > 1 else ' ' + list(missing_cols)[0], correlates_file
                                ))
                        else:
                            assert columns == required_columns, 'Columns must be provided in the same order for all correlates tsv files.'

                else:
                    assert len(line) == numcols, 'Record {}\non line {} has inconsistent number of columns. Expected {} columns, got {}.'.format(
                            txt, lineno, numcols, len(line)
                        )
                        
                    if any([f == '.' for f in line]):
                        raise ValueError('A value was not recorded for window {}: {}'
                                         'If this entry was made by "bedtools map", this means that the genomic feature file did not cover all'
                                         ' of the windows. Consider imputing a feature value or removing those windows.'.format(
                                             lineno, txt
                                         )
                                        )
                    try:
                        
                        correlates.append(
                            [float(f) for f in line] + [1.]
                        )

                    except ValueError as err:
                        raise ValueError('Could not ingest line {}: {}'.format(lineno, txt)) from err

        assert len(correlates) == len(windows), 'The number of correlates provided in {} does not match the number of specified windows.\n'\
                'Each window must have correlates provided.'

        correlates = np.array(correlates)

        return correlates, columns + ['constant']


    @staticmethod
    def read_windows(regions_file, genome_object, sep = '\t'):
        
        logger.info('Reading windows ...')
        
        windows = []
        with open(regions_file, 'r') as f:
            
            for lineno, txt in enumerate(f):

                if not txt[0] == '#':
                
                    line = txt.strip().split(sep)
                    assert len(line) >= 3, 'Bed files have three or more columns: chr, start, end, ...'

                    if len(line) > 4:
                        logger.warn(
                            'The only information ingested from the "regions_file" is the chr, start, end, and name of genomic bins. All other fields are ignored.'
                        )

                    try:
                        
                        windows.append(
                            Region(*line[:3], annotation= {'name' : line[3]} if len(line) > 3 else None)
                        )
                        
                    except ValueError as err:
                        raise ValueError('Could not ingest line {}: {}'.format(lineno, txt)) from err
                    

        last_w = windows[0]
        for w in windows[1:]:
            assert w.chromosome > last_w.chromosome or (w.chromosome == last_w.chromosome) and w.start > last_w.start,\
                'Provided windows must be in sorted order! Run "sort -k1,1 -k2,2n <windows>" to sort them.\n'\
                f'Last window: {last_w.chromosome}-{last_w.start}, next window: {w.chromosome}-{w.start}\n'\
                'IF YOU HAVE ALREADY MADE CORRELATES AND EXPOSURES TSVs, DO NOT FORGET TO REORDER THOSE SO THAT THEY MATCH WITH THE CORRECT WINDOWS!'

        return windows



    @staticmethod
    def collect_vcfs(vcf_files, sep = '\t', index = -1, chr_prefix = '',*,
                     fasta_file, regions_file):
        
        logger.info('Reading VCF files ...')
        samples = []
        for vcf in vcf_files:
            
            logger.info('Featurizing {}'.format(vcf))
            
            samples.append(
                Sample.featurize_mutations(vcf, regions_file = regions_file, fasta_file = fasta_file,
                                          sep = sep, index = index, chr_prefix = chr_prefix)
            )
        
        logger.info('Done reading VCF files.')
        return samples


    @staticmethod
    def calculate_exposures(windows, exposures):
        return np.array([[len(w)/10000 for w in windows]]) * exposures


    @staticmethod
    def collect_exposures(exposure_files, windows):
        
        logger.info('Reading exposure files ...')

        return np.vstack([
            CorpusReader.read_exposure_file(f, windows)[None,:]
            for f in exposure_files
        ])


    @staticmethod
    def get_trinucleotide_distributions(window_set, fasta_object, n_jobs = 1):
        
        def count_trinucs(chrom, start, end, fasta_object):

            def rolling(seq, w = 3):
                for i in range(len(seq) - w + 1):
                    yield seq[ i : i+w]

            window_sequence = fasta_object[chrom][start : end].seq.upper()
                
            trinuc_counts = Counter([
                trinuc for trinuc in rolling(window_sequence) if not 'N' in trinuc
            ])

            return [trinuc_counts[context] + trinuc_counts[revcomp(context)] for context in CONTEXT_IDX.keys()]
        
        trinuc_matrix = [
            count_trinucs(w.chromosome, w.start, w.end, fasta_object) 
            for w in tqdm.tqdm(window_set, nrows=30, desc = 'Aggregating trinucleotide content')
        ]

        trinuc_matrix = np.array(trinuc_matrix) + 1 # add a pseudocount

        return trinuc_matrix/trinuc_matrix.sum(1, keepdims = True)