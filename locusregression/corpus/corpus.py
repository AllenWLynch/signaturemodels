
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


class Dataset:

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __init__(self, 
            sep = '\t', index = -1, n_jobs = 1,*,
            fasta_file,
            genome_file,
            bedgraph_mtx,
            vcf_files,
        ):

        genome = Genome.from_file(genome_file)


        with Fasta(fasta_file) as fa:

            self._windows, self._features, self._feature_names = self.read_bedgraph(
                bedgraph_mtx, genome, fa, sep = sep
            )

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
            

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        
    @staticmethod
    def read_bedgraph(bedgraph_mtx, genome_object, fasta_object, sep = '\t'):
        
        logger.info('Reading genomic features ...')
        
        windows = []
        columns = []
        numcols = None
        with open(bedgraph_mtx, 'r') as f:
            
            for lineno, txt in enumerate(f):
                
                line = txt.strip().split(sep)

                if numcols is None:
                    numcols = len(line)
                    if not txt[0] == '#':
                        logger.warn(
                            'The first line of the bedgraph matrix file does not start with colnames prefixed with "#".\n'
                            'e.g. #chr\t#start\t#end\t#feature1 ... etc.'
                            'The first line of the file will be treated as column names.'
                            )

                    assert len(line) > 3, 'Cannot run algorithm without at least one feature provided.'
                    columns = [col.strip('#') for col in columns[3:]] + ['Constant']

                elif line[0] in fasta_object:
                    assert len(line) == numcols, 'Record {}\non line {} has inconsistent number of columns. Expected {} columns, got {}.'.format(
                            txt, lineno, numcols, len(line)
                        )
                        
                    if any([f == '.' for f in line[3:]]):
                        raise ValueError('A value was not recorded for window {}: {}'
                                         'If this entry was made by "bedtools map", this means that the genomic feature file did not cover all'
                                         ' of the windows. Consider imputing a feature value or removing those windows.'.format(
                                             lineno, txt
                                         )
                                        )
                    try:
                        
                        windows.append(
                            Region(*line[:3], annotation= {'features' : [
                                float(x) for x in line[3:]] + [1.]
                            }) # add the features of the bedgraph matrix to the annotation for that window
                        )
                        
                    except ValueError as err:
                        raise ValueError('Could not ingest line {}: {}'.format(lineno, txt)) from err

        windows = RegionSet(windows, genome_object)

        features = np.array([
            w.annotation['features'] for w in windows
        ])
        
        features = MinMaxScaler().fit_transform(features)

        return windows, features, columns


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
        return np.array([[len(w)/10000 for w in self._windows]])

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

    def __iter__(self):
        for samp in self.samples:

            yield {
                **samp, 
                'shared_correlates' : False,
                'window_size' : self.window_sizes, 
                'X_matrix' : self.genome_features,
                'trinuc_distributions' : self.trinucleotide_distributions,
            }

    def __len__(self):
        return len(self.samples)


    def _get(self, index):
        return {
                **self.samples[index], 
                'shared_correlates' : False,
                'window_size' : self.window_sizes, 
                'X_matrix' : self.genome_features,
                'trinuc_distributions' : self.trinucleotide_distributions,
            }


    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            return [self._get(g) for g in index]

        else:
            return self._get(index)
        



