
from .featurization import CONTEXT_IDX
from .corpus import Corpus, InMemorySamples
from pyfaidx import Fasta
import numpy as np
from collections import Counter
import logging
import tqdm
from .sbs_sample import revcomp, SBSSample
from .make_windows import check_regions_file
import os
from joblib import Parallel, delayed
from functools import partial
from dataclasses import dataclass
from dataclasses import dataclass
logger = logging.getLogger('DataReader')
logger.setLevel(logging.INFO)

   
@dataclass
class BED12Record:
    chromosome: str
    start: int
    end: int
    name: str
    score: int
    strand: str
    thick_start: int
    thick_end: int
    item_rgb: str
    block_count: int
    block_sizes: list[int]
    block_starts: list[int]

    def segments(self):
        for start, size in zip(self.block_starts, self.block_sizes):
            yield self.chromosome, self.start + start, self.start + start + size
        
    def __len__(self):
        return sum(self.block_sizes)



class CorpusReader:

    corpus_class = Corpus

    @classmethod
    def create_trinuc_file(cls,
            fasta_file,
            regions_file,
            output,
            n_jobs = 1,
        ):

        windows = cls.read_windows(regions_file, None, sep = '\t')

        trinuc_distributions = cls.get_trinucleotide_distributions(
                    windows, fasta_file, n_jobs=n_jobs
                )
            
        np.savez(output, x = trinuc_distributions)



    @classmethod
    def create_corpus(cls, 
            weight_col = None,
            chr_prefix = '', 
            exposure_files = None,
            trinuc_file = None,
            n_jobs=1,*,
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
            

        windows = cls.read_windows(regions_file, None, sep = '\t')

        if len(exposure_files) > 0:
            exposures = cls.collect_exposures(exposure_files, windows)
        else:
            exposures = np.ones((1, len(windows)))
        
        correlates_dict = cls.read_correlates(
                            correlates_file, 
                            required_columns=None
                         )
        
        assert len(next(iter(correlates_dict.values()))['values']) == len(windows), \
                'The number of correlates provided in {} does not match the number of specified windows.\n'\
                'Each window must have correlates provided.'

        samples = cls.collect_vcfs(
            weight_col = weight_col,
            vcf_files = vcf_files, 
            fasta_file = fasta_file, 
            regions_file = regions_file,
            chr_prefix = chr_prefix,
            exposures = exposures,
            n_jobs = n_jobs,
        )

        trinuc_distributions = None
        if trinuc_file is None:
            trinuc_distributions = cls.get_trinucleotide_distributions(
                        windows, fasta_file, n_jobs=n_jobs
                    )    
        else:
            trinuc_distributions = cls.read_trinuc_distribution(trinuc_file)

            assert trinuc_distributions.shape[0] == len(windows), \
                'The number of trinucleotide distributions provided in {} does not match the number of specified windows.\n'\

        shared = exposures.shape[0] == 1
        if shared:
            logger.info('Sharing exposures between samples.')


        return Corpus(
            samples = InMemorySamples(samples),
            features = correlates_dict,
            trinuc_distributions = trinuc_distributions.T,
            shared_exposures = shared,
            name = corpus_name,
            metadata={
                'regions_file' : os.path.abspath(regions_file),
                'fasta_file' : os.path.abspath(fasta_file),
                'correlates_file' : os.path.abspath(correlates_file),
                'trinuc_file' : os.path.abspath(trinuc_file) if not trinuc_file is None else None,
                'chr_prefix' : str(chr_prefix),
                'weight_col' : str(weight_col),
            }
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
    def read_correlates(correlates_file, 
        required_columns = None, sep = '\t'):
        
        logger.info('Reading genomic features ...')

        type_map = {
            'categorical' : str
        }
        
        correlates = []
        with open(correlates_file, 'r') as f:
            
            cols = next(f).strip().split(sep)
            assert len(cols) > 0
            numcols = len(cols)
            
            if not all(col.startswith('#feature=') for col in cols):
                raise ValueError('The first line of the tsv file must be columns of "#feature=" followed by the feature name.\n'
                                    'e.g. #feature=H3K27ac\n'
                                    'The first line of the file will be treated as column names.'
                                )

            feature_names = [col.removeprefix('#feature=') for col in cols]

            if required_columns is None:
                logger.info('Using correlates: ' + ', '.join(feature_names))
            else:
                missing_cols = set(required_columns).difference(feature_names)

                if len(missing_cols) > 0:
                    raise ValueError('The required correlate {} was missing from file: {}.\n All correlates must be specified for all samples'.format(
                            's ' + ', '.join(missing_cols) if len(missing_cols) > 1 else ' ' + list(missing_cols)[0], correlates_file
                        ))

            cols = next(f).strip().split(sep)
            if not all(col.startswith('#type=') for col in cols):
                raise ValueError('The second line of the tsv file must be columns of "#type=" followed by the feature type.\n'
                                    'e.g. #type=power\n'
                                    'The second line of the file will be treated as column names.'
                                )
            
            feature_types= [col.removeprefix('#type=') for col in cols]
            
            #if not all(t in ['continuous', 'discrete', 'distance'] for t in feature_types):
            #    logger.warn('Found invalid types. The feature type must be either "continuous", "discrete", or "distance" for automatic normalization.')
            
            cols = next(f).strip().split(sep)
            if not all(col.removeprefix('#group=') for col in cols):
                raise ValueError('The third line of the tsv file must be columns of "#group=" followed by the feature group.\n'
                                    'e.g. #group=replication timing\n'
                                    'The third line of the file will be treated as column names.'
                                )
            
            groups = [col.strip('#group=') for col in cols]
            
            for lineno, txt in enumerate(f):
                lineno+=3

                line = txt.strip().split(sep)
                assert len(line) == numcols, 'Record {}\non line {} has inconsistent number of columns. Expected {} columns, got {}.'.format(
                        txt, lineno, numcols, len(line)
                    )
                    
                if any([f == '.' and not t=='categorical' for f,t in zip(line, feature_types)]):
                    logger.warn('A value was not recorded for window {}: {} for a continous feature.'
                                        'If this entry was made by "bedtools map", this means that the genomic feature file did not cover all'
                                        ' of the windows.'.format(
                                            lineno, txt
                                        )
                                    )
                try:
                    # convert each feature in the line to the appropriate type
                    # if the feature is categorical, it will be left as a string
                    # otherwise, it will be converted to a float.
                    # If "." is encountered, it will be converted to float(nan) for numerical features 
                    # and left as a string for categorical features.
                    correlates.append(
                        [type_map.setdefault(_type, float)(_feature) if not _feature=='.' else type_map.setdefault(_type, float)('nan')
                         for _feature,_type in zip(line, feature_types)
                        ]
                    )

                except ValueError as err:
                    raise ValueError('Could not ingest line {}: {}'.format(lineno, txt)) from err

        correlates_dict = {}
        for name, _type, group, vals in zip(
            feature_names, feature_types, groups, zip(*correlates)
        ):
            
            correlates_dict[name] = {
                'type' : _type,
                'group' : group,
                'values' : np.array(vals).astype(type_map.setdefault(_type,float)),
            }
            
        return correlates_dict
    

    @staticmethod
    def read_windows(regions_file, genome_object, sep = '\t'):

        def parse_bed12_line(line):
            chromosome, start, end, name, score, strand, thick_start, thick_end, item_rgb, block_count, block_sizes, block_starts = line.strip().split('\t')
            block_sizes = list(map(int, block_sizes.split(',')))
            block_starts = list(map(int, block_starts.split(',')))
            
            return BED12Record(
                chromosome=chromosome,
                start=int(start),
                end=int(end),
                name=name,
                score=int(score),
                strand=strand,
                thick_start=int(thick_start),
                thick_end=int(thick_end),
                item_rgb=item_rgb,
                block_count=int(block_count),
                block_sizes=block_sizes,
                block_starts=block_starts
            )

        check_regions_file(regions_file)
        
        logger.info('Reading windows ...')
        
        windows = []
        with open(regions_file, 'r') as f:
            
            for lineno, txt in enumerate(f):
                if not txt[0] == '#':
                    line = txt.strip().split(sep)
                    assert len(line) >= 12, 'Expected BED12 file type with at least 12 columns'

                try:
                    windows.append(
                        parse_bed12_line(txt)
                    )
                except ValueError as err:
                    raise ValueError('Could not ingest line {}: {}'.format(lineno, txt)) from err

        return windows



    @staticmethod
    def collect_vcfs(vcf_files, 
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

            _exposures = duplication_iterator(len(vcf_files))

        else:
            _exposures = list(exposures)

            assert len(_exposures) == len(vcf_files), \
                'If providing exposures, provide one for the whole corpus, or one per sample.'

        read_vcf_fn = partial(
            SBSSample.featurize_mutations,
            regions_file = regions_file, 
            fasta_file = fasta_file,
            chr_prefix = chr_prefix, 
            weight_col = weight_col
        )

        samples = Parallel(
                        n_jobs = n_jobs, 
                        verbose = 10,
                    )(
                        delayed(read_vcf_fn)(vcf, exposures = sample_exposure)
                        for vcf, sample_exposure in zip(vcf_files, _exposures)
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
    def get_trinucleotide_distributions(window_set, fasta_file, n_jobs = 1):
        
        def count_trinucs(bed12_region, fasta_object):

            def rolling(seq, w = 3):
                for i in range(len(seq) - w + 1):
                    yield seq[ i : i+w]

            trinuc_counts = Counter()

            for chrom, start, end in bed12_region.segments():

                window_sequence = fasta_object[chrom][max(start-1,0) : end+1].seq.upper()
                    
                trinuc_counts += Counter([
                    trinuc for trinuc in rolling(window_sequence) if not 'N' in trinuc
                ])

            return [trinuc_counts[context] + trinuc_counts[revcomp(context)] for context in CONTEXT_IDX.keys()]
        
        with Fasta(fasta_file) as fasta_object:
            trinuc_matrix = [
                count_trinucs(w, fasta_object) 
                for w in tqdm.tqdm(window_set, nrows=30, desc = 'Aggregating trinucleotide content')
            ]

        trinuc_matrix = np.array(trinuc_matrix) # DON'T (!) add a pseudocount

        return trinuc_matrix # DON'T (!) normalize, the number of contexts in a window is part of the likelihood
    

    @classmethod
    def ingest_sample(cls, vcf_file, exposure_file = None,*,
                    regions_file,
                    fasta_file,
                    chr_prefix = '',
                    weight_col = None,
                    ):
        
        windows = cls.read_windows(regions_file, None, sep = '\t')

        if not exposure_file is None:
            exposures = cls.collect_exposures(exposure_file, windows)
        else:
            exposures = np.ones((1, len(windows)))

        logger.info('Ingesting mutations ...')
        return SBSSample.featurize_mutations(
            vcf_file, 
            regions_file,
            fasta_file,
            exposures,
            chr_prefix = chr_prefix,
            weight_col = weight_col,
        )