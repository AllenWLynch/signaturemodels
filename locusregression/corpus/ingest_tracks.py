import subprocess
import tempfile
import logging
from .make_windows import check_regions_file
logger = logging.getLogger('Roadmap downloader')
logger.setLevel(logging.INFO)


def process_bigwig(*,bigwig_file, regions_file, output,
                   feature_name = None):
    
    check_regions_file(regions_file)

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
