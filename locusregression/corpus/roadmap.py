import subprocess
from multiprocessing.pool import ThreadPool as Pool
import tempfile
import sys
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

def fetch_roadmap_features(*,roadmap_id, windows_file, output, n_jobs = 1):

    try:

        aggregated_files = {
            signal : tempfile.NamedTemporaryFile()
            for signal in config.keys()
        }

        def _summarize_bigwig(signal, track_type, output_file):

            with tempfile.NamedTemporaryFile() as bigwig, \
                    tempfile.NamedTemporaryFile() as bed:

                _url = ROADMAP_URL.format(
                    signal = signal,
                    sample = roadmap_id,
                    type = track_type
                )

                logger.info(f'Downloading from {_url}')
                subprocess.check_output(['curl', _url, '-o', bigwig.name, '-f', '-s'])

                subprocess.check_output(
                    ['bigWigAverageOverBed', bigwig.name, windows_file, bed.name],
                )

                subprocess.Popen(
                    ['cut','-f5', bed.name],
                    stdout = output_file
                ).communicate()


        with Pool(n_jobs) as pool:

            for signal, track_type in config.items():
                pool.apply_async(
                    _summarize_bigwig,
                    args = (signal, track_type, aggregated_files[signal])
                )

            pool.close()
            pool.join()

        logger.info(f'Merging features ...')
        print(*['#' + mark for mark in config.keys()], 
              sep = '\t', file = output)
        
        output.flush()

        subprocess.Popen(
            ['paste', *[f.name for f in aggregated_files.values()]],
            stdout = output
        ).communicate()

    finally:
        for f in aggregated_files.values():
            f.close()



        

