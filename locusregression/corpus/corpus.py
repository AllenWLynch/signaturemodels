
from pyfaidx import Fasta
from .genome_tools import Region, RegionSet, Genome
from .featurization import CONTEXT_IDX, MUTATIONS_IDX
import numpy as np
from collections import Counter
import logging
import tqdm
from joblib import Parallel, delayed
import pickle
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger('Corpus')
logger.setLevel(logging.INFO)

#
# SETTING UP MUTATION INDICES
#
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


class Sample:

    @staticmethod
    def _read_mutations(vcf_file, genome_object, 
                        sep = '\t', index = -1):

        sbs_mutations = []
        all_sbs = 0
        
        with open(vcf_file, 'r') as f:
            
            for line in f:
                if not line[0] == '#':
                    line = line.strip().split(sep)
                    #assert len(line) == 12, 'This is not a valid VCF file. It should have twelve columns'

                    if line[VCF.REF] in 'ATCG' and line[VCF.ALT] in 'ATCG': # check for SBS
                        
                        newregion = SbsRegion(
                                'chr' + line[VCF.CHROM],int(line[VCF.POS])-1+index, int(line[VCF.POS])+2+index, # add some buffer indices for the context
                                annotation=(line[VCF.REF], line[VCF.ALT])
                            )
                        
                        all_sbs += 1
                        if genome_object.contains_region(newregion):
                            sbs_mutations.append(newregion)
                        
        
        
        if all_sbs == 0:
            logger.warn('VCF file {} contained no valid SBS mutations.')
        elif len(sbs_mutations)/all_sbs < 0.5:
            logger.warn('It appears there was a mismatch between the names of chromosomes in your VCF file and genome file.'
                       'Many mutations were rejected.')
                
        return RegionSet(sbs_mutations, genome_object)


    @staticmethod
    def _get_context_mutation_idx(fasta_object, sbs):

        try:
            context = fasta_object[sbs.chromosome][sbs.start : sbs.end].seq.upper()
        except KeyError as err:
            raise KeyError('Chromosome {} found in VCF file is not in the FASTA reference file'\
                .format(sbs.chromosome)) from err

        newref = context[1]

        assert newref == sbs.ref, 'Looks like the vcf file was constructed with a different reference genome, different ref allele found at {}:{}, found {} instead of {}'.format(
            sbs.chromosome, str(sbs.start+1), newref, sbs.ref 
        )

        #print(context, sbs.ref, sbs.alt)
        context, alt = convert_to_mutation(context, sbs.alt)

        return MUTATIONS_IDX[context][alt], CONTEXT_IDX[context]


    @staticmethod
    def _collect_windows(sbs_set, window_set):

        window_intersections = sbs_set.map_intersects(window_set)
        
        window_intersections = sparse.hstack([
            sparse.csr_matrix((window_intersections.shape[0],1)),
            window_intersections
        ]).argmax(1) - 1
        
        return np.ravel(window_intersections)


    @staticmethod
    def _aggregate_counts(mutation_indices, context_indices, locus_indices):

        aggregated = Counter(zip(mutation_indices, context_indices, locus_indices))

        mutations, contexts, loci = list(zip(*aggregated.keys()))
        counts = list(aggregated.values())

        return {
            'mutation' : np.array(mutations), 
            'context' : np.array(contexts),
            'locus' : np.array(loci), 
            'count' : np.array(counts)
        }

    @staticmethod
    def featurize_mutations(vcf, fasta_object, genome_object, window_set, 
        sep = '\t', index= -1):

        sbs_set = Sample._read_mutations(vcf, genome_object, sep = sep, index = index)

        mutation_indices, context_indices = list(zip(*[
            Sample._get_context_mutation_idx(fasta_object, sbs) for sbs in sbs_set.regions
        ]))

        locus_indices = Sample._collect_windows(sbs_set, window_set)
        
        unmatched_mask = locus_indices == -1
        
        if unmatched_mask.sum() > 0:
            logger.warn('{} mutations did not intersect with any provided windows, filtering these out.')

        return Sample._aggregate_counts(
            np.array(mutation_indices)[~unmatched_mask], 
            np.array(context_indices)[~unmatched_mask], 
            np.array(locus_indices)[~unmatched_mask]
        )


class BaseCorpus:

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump([x for x in self], f)


class Corpus(BaseCorpus):

    def __init__(self, 
            sep = '\t', 
            index = -1, 
            n_jobs = 1,
            exposure_file = None,*,
            correlates_file,
            fasta_file,
            genome_file,
            vcf_files,
            regions_file,
        ):

        genome = Genome.from_file(genome_file)

        self._windows = self.read_windows(regions_file, genome, sep = sep)

        self._exposures = self.calculate_exposures(
            self._windows, 
            self.read_exposure_file(exposure_file, self._windows, sep = sep) \
                if not exposure_file is None else np.ones(len(self._windows))
        )
        
        self._features, self._feature_names = self.read_correlates(
            correlates_file, self._windows, required_columns=None)


        with Fasta(fasta_file) as fa:

            self._samples = self.collect_vcfs(
                vcf_files = vcf_files, 
                fasta_object = fa, 
                genome_object = genome,
                window_set = self._windows,
                index = index,
                sep = sep,
            )

            self._trinuc_distributions = self.get_trinucleotide_distributions(
                self._windows, fa, n_jobs=n_jobs
            )


    @staticmethod
    def read_exposure_file(exposure_file, windows):
        
        exposures = []
        with open(exposure_file, 'r') as f:

            for lineno, txt in enumerate(f):

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
                            raise ValueError('The required correlate{} was missing from file: {}.\n All correlates must be specified for all samples'.format(
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

        correlates = MinMaxScaler().fit_transform(correlates)

        return correlates, columns + ['constant']


    @staticmethod
    def read_windows(regions_file, genome_object, sep = '\t'):
        
        logger.info('Reading windows ...')
        
        windows = []
        with open(regions_file, 'r') as f:
            
            for lineno, txt in enumerate(f):
                
                line = txt.strip().split(sep)
                assert len(line) >= 3, 'Bed files have three or more columns: chr, start, end, ...'

                if len(line) > 4:
                    logger.warn(
                        'The only information ingested from the "regions_file" is the chr, start, end, and name of genomic bins. All other fields are ignored.'
                    )

                try:
                    
                    windows.append(
                        Region(*line[:3], annotation= {'name' : line[4]} if len(line) > 3 else None)
                    )
                    
                except ValueError as err:
                    raise ValueError('Could not ingest line {}: {}'.format(lineno, txt)) from err

        windows = RegionSet(windows, genome_object)

        return windows


    @staticmethod
    def collect_vcfs(vcf_files, sep = '\t', index = -1,*,
                     fasta_object, genome_object, window_set):
        
        logger.info('Reading VCF files ...')
        samples = []
        for vcf in vcf_files:
            
            logger.info('Featurizing {}'.format(vcf))
            
            samples.append(
                Sample.featurize_mutations(vcf, fasta_object, genome_object, window_set,
                                          sep = sep, index = index)
            )
        
        logger.info('Done reading VCF files.')
        return samples


    @staticmethod
    def calculate_exposures(windows, exposures):
        return np.array([[len(w)/10000 for w in windows]]) * exposures[None,:]


    @staticmethod
    def get_trinucleotide_distributions(window_set, fasta_object, n_jobs = 1):
    
        return np.ones((len(window_set), 32))/32
        
        def count_trinucs(chrom, start, end, fasta_object):

            def rolling(seq, w = 3):
                for i in range(len(seq) - w + 1):
                    yield seq[ i : i+w]

            window_sequence = fasta_object[chrom][start : end].seq.upper()
                
            trinuc_counts = Counter([
                trinuc for trinuc in rolling(window_sequence) if not 'N' in trinuc
            ])

            return [trinuc_counts[context] + trinuc_counts[revcomp(context)] for context in CONTEXT_IDX.keys()]
        
        trinuc_matrix = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(count_trinucs)(w.chromosome, w.start, w.end, fasta_object) 
            for w in tqdm.tqdm(window_set, nrows=30, desc = 'Aggregating trinucleotide content')
        )

        return np.array(trinuc_matrix)


    @property
    def window_sizes(self):
        return self._exposures

    @property
    def exposures(self):
        return self._exposures

    @property
    def samples(self):
        return self._samples

    @property
    def windows(self):
        return self._windows

    @property
    def genome_features(self):
        return self._features.T

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def trinucleotide_distributions(self):
        return self._trinuc_distributions.T

    @property
    def mutations(self):
        return [sample['mutation'] for sample in self._samples]
    
    @property
    def contexts(self):
        return [sample['context'] for sample in self._samples]

    @property
    def loci(self):
        return [sample['locus'] for sample in self._samples]

    @property
    def counts(self):
        return [sample['count'] for sample in self._samples]

    def _get(self, index):
        return {
                **self.samples[index], 
                'shared_correlates' : True,
                'window_size' : self.exposures, 
                'X_matrix' : self.genome_features,
                'trinuc_distributions' : self.trinucleotide_distributions,
            }

    def __iter__(self):
        for i in range(len(self)):
            yield self._get(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            return [self._get(g) for g in index]

        else:
            return self._get(index)


class MixedCorpus(Corpus):
    
    def __init__(self, 
            sep = '\t', index = -1, n_jobs = 1,*,
            correlates_files,
            exposure_files,
            vcf_files,
            fasta_file,
            genome_file,
            regions_file,
        ):

        assert len(correlates_files) == len(vcf_files), 'User must provide a VCF file, correlates file, and exposure file for each sample.'

        assert len(exposure_files) in [0,1,len(vcf_files)]

        genome = Genome.from_file(genome_file)
        self._windows = self.read_windows(regions_file, genome, sep = sep)

        if len(exposure_files) == 0:
            exposures = np.ones((len(vcf_files), len(self._windows)))
        elif len(exposure_files) == 1:
            exposures = np.repeat(
                self.read_exposure_file(exposure_files[0], self._windows),
                len(vcf_files), axis = 0
            )
        else:
            exposures = self.collect_exposures(exposure_files, self._windows)


        self._exposures = self.calculate_exposures(
            self._windows, exposures
        )
        
        self._features, self._feature_names = self.collect_correlates(
            correlates_files, self._windows, 
            required_columns=None, 
            sep = sep,
        )

        with Fasta(fasta_file) as fa:

            self._samples = self.collect_vcfs(
                vcf_files=vcf_files, 
                fasta_object=fa, 
                genome_object=genome,
                window_set= self._windows,
                index = index,
                sep = sep,
            )

            self._trinuc_distributions = self.get_trinucleotide_distributions(
                self._windows, fa, n_jobs=n_jobs
            )


    @staticmethod
    def calculate_exposures(windows, exposures):
        return np.array([[len(w)/10000 for w in windows]]) * exposures


    @staticmethod
    def collect_exposures(exposure_files, windows):

        return np.vstack([
            MixedCorpus.read_exposure_file(f, windows)[None,:]
            for f in exposure_files
        ])


    @staticmethod
    def collect_correlates(correlates_files, windows, sep = '\t'):

        columns = None
        correlates = []

        for corrs in correlates_files:
            new_correlates, columns = MixedCorpus.read_correlates(corrs, windows, 
                required_columns=columns, sep = sep)

            correlates.append(new_correlates)

        return correlates


    def _get(self, index):
        return {
                **self.samples[index], 
                'window_size' : self.exposures[index], 
                'X_matrix' : self.genome_features[index],
                'trinuc_distributions' : self.trinucleotide_distributions,
                'shared_correlates' : False,
            }