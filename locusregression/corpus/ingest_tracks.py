import subprocess
from multiprocessing.pool import ThreadPool as Pool
import tempfile
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
import logging
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

def _check_windows_file(windows_file):

    with open(windows_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            assert len( line.strip().split('\t') ) == 4, \
                f'Expected 4 columns in {windows_file}'
        

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



def process_bigwig(*,bigwig_file, windows_file, output,
                   feature_name = None, normalization = None):
    
    _check_windows_file(windows_file)

    with tempfile.NamedTemporaryFile() as bed, \
        tempfile.NamedTemporaryFile() as track:

        subprocess.check_output(
                        ['bigWigAverageOverBed', bigwig_file, windows_file, bed.name],
                    )

        cut_process = subprocess.Popen(
            ['cut','-f5', bed.name],
            stdout = track,
        )
        cut_process.communicate()
        track.flush()

        if not cut_process.returncode == 0:
            raise RuntimeError('Command returned non-zero exit status')

        normalize_covariate(track_file=track.name, 
                            output=output,
                            normalization=normalization,
                            feature_name=feature_name
                            )


def process_bedgraph(*,bedgraph_file, windows_file, output,
                     column = 4, feature_name = None, normalization = None):
    
    _check_windows_file(windows_file)
    
    with tempfile.NamedTemporaryFile() as track:

        map_process = subprocess.Popen(
            ['bedtools', 'map','-a', windows_file, '-b', bedgraph_file, 
            '-c', str(column), '-o', 'mean', 
            '-sorted', '-null', '0.'],
            stdout = subprocess.PIPE,
            bufsize=1000,
            universal_newlines=True,
        )

        cut_process = subprocess.Popen(
            ['cut','-f5'],
            stdin = map_process.stdout,
            stdout = track,
        )

        cut_process.communicate(); map_process.communicate()
        track.flush()

        if not cut_process.returncode == 0 or not map_process.returncode == 0:
            raise RuntimeError('Command returned non-zero exit status')

        normalize_covariate(track_file=track.name, 
                            output=output,
                            normalization=normalization,
                            feature_name=feature_name
                            )
    


def fetch_roadmap_features(*,roadmap_id, windows_file, output, n_jobs = 1):

    try:

        aggregated_files = {
            signal : tempfile.NamedTemporaryFile()
            for signal in config.keys()
        }

        def _summarize_bigwig(signal, track_type, output_file):

            with tempfile.NamedTemporaryFile() as bigwig, \
                    tempfile.NamedTemporaryFile() as bed, \
                    tempfile.NamedTemporaryFile() as track:

                _url = ROADMAP_URL.format(
                    signal = signal,
                    sample = roadmap_id,
                    type = track_type
                )

                logger.info(f'Downloading from {_url}')
                subprocess.check_output(['curl', _url, '-o', bigwig.name, '-f', '-s'])

                with open(output_file.name, 'w') as f:
            
                    process_bigwig(
                        bigwig_file = bigwig.name,
                        windows_file = windows_file,
                        output = f,
                        feature_name = signal,
                        normalization = 'power',
                    )


        for signal, track_type in config.items():
            _summarize_bigwig(signal, track_type, aggregated_files[signal])

        '''with Pool(n_jobs) as pool:

            for signal, track_type in config.items():
                pool.apply_async(
                    _summarize_bigwig,
                    args = (signal, track_type, aggregated_files[signal])
                )

            pool.close()
            pool.join()'''

        logger.info(f'Merging features ...')

        subprocess.Popen(
            ['paste', *[f.name for f in aggregated_files.values()]],
            stdout = output
        ).communicate()

    finally:
        for f in aggregated_files.values():
            f.close()



        

