import subprocess
import tempfile
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
import logging
import os
logger = logging.getLogger('Roadmap downloader')
logger.setLevel(logging.INFO)

config = {
    'RNAseq' : 'LogRPKM',
    'H3K27ac' : 'pval',
    'DNAMethylSBS' : 'FractionalMethylation',
    'H3K4me1' : 'pval',
    'H3K4me3' : 'pval',
    'H3K9me3' : 'pval',
    'H3K27me3' : 'pval',
    'H3K36me3' : 'pval',
}


ROADMAP_URL = 'https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidatedImputed/{signal}/{sample}-{signal}.imputed.{type}.signal.bigwig'
BIGWIG_NAME = '{sample}-{signal}.imputed.{type}.signal.bigwig'


def _check_regions_file(regions_file):

    with open(regions_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            assert len( line.strip().split('\t') ) == 4, \
                f'Expected 4 columns in {regions_file}, add a column to the regions file using \'awk -v OFS="\\t" \'{{print $0,NR}}\' <filename>\''
        

def normalize_covariate(*,track_file, output, feature_name,
                        normalization = None):
    
    with open(track_file) as track:
        track = track.readlines()
        track = np.array( [float(x.strip()) for x in track] )[:,np.newaxis]

    if normalization == 'power':
        track = PowerTransformer(standardize = False).fit_transform(track)
        track = MinMaxScaler().fit_transform(track)
    elif normalization == 'standardize':
        track = StandardScaler().fit_transform(track)
    elif normalization == 'minmax':
        track = MinMaxScaler().fit_transform(track)
    else:
        raise ValueError(f'Unknown normalization: {normalization}')

    track = list(track.astype(np.float32).reshape(-1))
    
    print('#' + feature_name, file= output)
    print(*track, sep = '\n', file= output)



def process_bigwig(*,bigwig_file, regions_file, output,
                   feature_name = None):
    
    _check_regions_file(regions_file)

    print('#' + feature_name, file= output, flush = True)

    with tempfile.NamedTemporaryFile() as bed:

        subprocess.check_output(
                        ['bigWigAverageOverBed', bigwig_file, regions_file, bed.name],
                    )

        sort_process = subprocess.Popen(
            ['sort', '-k1,1n', bed.name],
            stdout = subprocess.PIPE,
        )

        cut_process = subprocess.Popen(
            ['cut','-f5'],
            stdin = sort_process.stdout,
            stdout = output,
        )
        cut_process.communicate()
        output.flush()

        if not cut_process.returncode == 0:
            raise RuntimeError('Command returned non-zero exit status')
        


def process_bedgraph(*,bedgraph_file, regions_file, output,
                     column = 4, feature_name = None):
    
    _check_regions_file(regions_file)

    print('#' + feature_name, file= output, flush = True)

    map_process = subprocess.Popen(
        ['bedtools', 'map','-a', regions_file, '-b', bedgraph_file, 
        '-c', str(column), '-o', 'mean', 
        '-sorted', '-null', '0.'],
        stdout = subprocess.PIPE,
        bufsize=1000,
        universal_newlines=True,
    )

    cut_process = subprocess.Popen(
        ['cut','-f5'],
        stdin = map_process.stdout,
        stdout = output,
    )

    cut_process.communicate(); map_process.communicate()
    output.flush()

    if not cut_process.returncode == 0 or not map_process.returncode == 0:
        raise RuntimeError('Command returned non-zero exit status')

    


def fetch_roadmap_features(*,roadmap_id, regions_file, output, 
                           bigwig_dir = 'bigwigs', n_jobs = 1):

    logger.info('Saving bigwigs to "{}" directory.'.format(bigwig_dir))

    if not os.path.isdir(bigwig_dir):
        os.makedirs(bigwig_dir)

    try:

        aggregated_files = {
            signal : tempfile.NamedTemporaryFile()
            for signal in config.keys()
        }

        def _summarize_bigwig(signal, track_type, output_file):

            bw_kw = dict(
                signal = signal,
                sample = roadmap_id,
                type = track_type
            )
            
            bigwig_file = os.path.join(bigwig_dir, BIGWIG_NAME.format(**bw_kw))

            if not os.path.isfile(bigwig_file):
                
                _url = ROADMAP_URL.format(**bw_kw)

                logger.info(f'Downloading from {_url}')
                
                try:
                    subprocess.check_output(['curl', _url, '-o', bigwig_file, '-f', '-s'])
                except (Exception, KeyboardInterrupt) as err:

                    logger.error(f'Failed to download {_url}, removing corrupted file.')
                    os.remove(bigwig_file)
                    raise RuntimeError(f'Failed to download {_url}') from err
            else:
                logger.info(f'Found {bigwig_file}, using downloaded version.')


            with open(output_file.name, 'w') as f:
        
                process_bigwig(
                    bigwig_file = bigwig_file,
                    regions_file = regions_file,
                    output = f,
                    feature_name = signal,
                )

        for signal, track_type in config.items():
            _summarize_bigwig(signal, track_type, aggregated_files[signal])


        logger.info(f'Merging features ...')

        subprocess.Popen(
            ['paste', *[f.name for f in aggregated_files.values()]],
            stdout = output
        ).communicate()

    finally:
        for f in aggregated_files.values():
            f.close()



        

