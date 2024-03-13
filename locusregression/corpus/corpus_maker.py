from .corpus import Corpus, InMemorySamples
from .make_windows import check_regions_file
import numpy as np
import logging
import os
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
    
    def __str__(self):
        return '\t'.join(
            map(str, [self.chromosome, self.start, self.end, 
                      self.name, self.score, self.strand, 
                      self.thick_start, self.thick_end, 
                      self.item_rgb, self.block_count, 
                      ','.join(map(str, self.block_sizes)), 
                      ','.join(map(str, self.block_starts))
                    ]
            )
        )


class CorpusMaker:

    corpus_type = 'SBS'

    @classmethod
    def create_context_frequencies_file(cls,
            fasta_file,
            regions_file,
            output,
            n_jobs = 1,
        ):

        windows = cls.read_windows(regions_file, sep = '\t')

        context_frequencies = cls.get_context_frequencies(
                    windows, fasta_file, n_jobs=n_jobs
                )
            
        np.savez(output, x = context_frequencies)

    @classmethod
    def create_corpus(cls, 
            weight_col = None,
            chr_prefix = '', 
            exposure_files = None,
            context_file = None,
            n_jobs=1,*,
            correlates_file,
            fasta_file,
            sample_files,
            regions_file,
            corpus_name,
        ):

        if exposure_files is None:
            exposure_files = []

        elif len(exposure_files) > 1:
            assert len(exposure_files) == len(sample_files), \
                'If providing exposure files, provide one for the whole corpus, or one per sample.'
            

        windows = cls.read_windows(regions_file, sep = '\t')

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

        samples = cls.collect_samples(
            weight_col = weight_col,
            sample_files = sample_files, 
            fasta_file = fasta_file, 
            regions_file = regions_file,
            chr_prefix = chr_prefix,
            exposures = exposures,
            n_jobs = n_jobs,
        )

        context_frequencies = None
        if context_file is None:
            context_frequencies = cls.get_context_frequencies(
                        windows, fasta_file, n_jobs=n_jobs
                    )    
        else:
            context_frequencies = cls.read_context_frequencies(context_file)

            assert context_frequencies.shape[-1] == len(windows), \
                'The number of trinucleotide distributions provided in {} does not match the number of specified windows.\n'\

        shared = exposures.shape[0] == 1
        if shared:
            logger.info('Sharing exposures between samples.')

        return Corpus(
            type=cls.corpus_type,
            samples = InMemorySamples(samples),
            features = correlates_dict,
            context_frequencies = context_frequencies,
            shared_exposures = shared,
            name = corpus_name,
            metadata={
                'regions_file' : os.path.abspath(regions_file),
                'fasta_file' : os.path.abspath(fasta_file),
                'correlates_file' : os.path.abspath(correlates_file),
                'context_file' : os.path.abspath(context_file) if not context_file is None else None,
                'chr_prefix' : str(chr_prefix),
                'weight_col' : str(weight_col) if not weight_col is None else 'None',
            }
        )
    
    @staticmethod
    def featurize_mutations(sample_file,
                            regions_file,
                            fasta_file,
                            exposures,
                            chr_prefix = '',
                            weight_col = None,
                            **kw):
        raise NotImplementedError('This method must be implemented by the subclass.')
    
    
    @staticmethod
    def collect_samples(sample_files,
                        fasta_file,
                        regions_file,
                        exposures,
                        chr_prefix = '',
                        weight_col = None,
                        n_jobs=1):
        raise NotImplementedError('This method must be implemented by the subclass.')
    

    @staticmethod
    def get_context_frequencies(windows, fasta_file, n_jobs=1):
        raise NotImplementedError('This method must be implemented by the subclass.')
    

    @classmethod
    def collect_exposures(cls, exposure_files, windows):
        
        logger.info('Reading exposure files ...')

        return np.vstack([
            cls.read_exposure_file(f, windows)[None,:]
            for f in exposure_files
        ])

    
    @staticmethod
    def read_context_frequencies(context_file):
        return np.load(context_file)['x']


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
            'categorical' : str,
            'cardinality' : str,
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
                
                null_values = [f == '.' and not t in ['categorical','cardinality'] for f,t in zip(line, feature_types)]

                if any(null_values):
                    logger.info('A value was not recorded for window {}: {} for a continous feature: {}'
                                        'If this entry was made by "bedtools map", this means that the genomic feature file did not cover all'
                                        ' of the windows.'.format(
                                            lineno, txt, feature_names[null_values.index(True)]
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
    def read_windows(regions_file, sep = '\t'):

        def parse_bed12_line(line):
            chromosome, start, end, name, score, strand, thick_start, thick_end, item_rgb, block_count, block_sizes, block_starts = line.strip().split('\t')
            block_sizes = list(map(int, block_sizes.split(',')))
            block_starts = list(map(int, block_starts.split(',')))
            
            return BED12Record(
                chromosome=chromosome,
                start=int(start),
                end=int(end),
                name=name,
                score=float(score),
                strand=strand,
                thick_start=int(thick_start),
                thick_end=int(thick_end),
                item_rgb=item_rgb,
                block_count=int(block_count),
                block_sizes=block_sizes,
                block_starts=block_starts
            )

        check_regions_file(regions_file)
        
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
    

    @classmethod
    def ingest_sample(cls, vcf_file, 
                    chr_prefix = '',
                    weight_col = None,
                    exposure_file = None,
                    mutation_rate_file=None,*,
                    regions_file,
                    fasta_file,
                    ):
            
        windows = cls.read_windows(regions_file, sep = '\t')

        if not exposure_file is None:
            exposures = cls.collect_exposures(exposure_file, windows)
        else:
            exposures = np.ones((1, len(windows)))

        return cls.featurize_mutations(
            vcf_file, 
            regions_file,
            fasta_file,
            exposures,
            chr_prefix = chr_prefix,
            weight_col = weight_col,
        )