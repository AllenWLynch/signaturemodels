
import subprocess
import tempfile
from math import log10

def get_rainfall_statistic(vcf_file,*,output):

    def get_distance_cutoff(mutations_file):
        return 10000

    with tempfile.NamedTemporaryFile() as temp_file:

        filter_process = subprocess.Popen(
                ['bcftools','view','-v','snps', vcf_file],
                stdout = subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
                stderr = subprocess.DEVNULL,
            )
        
        query_process = subprocess.Popen(
            ['bcftools','query','-f', '%CHROM\t%POS0\t%POS0' + '\n',],
            stdin = filter_process.stdout,
            stdout = subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )
        
        with open(temp_file.name,'w') as f:
            sort_process = subprocess.Popen(
                ['sort','-k1,1','-k2,2n'],
                stdin = query_process.stdout,
                stdout = f,
                universal_newlines=True,
                bufsize=10000,
            )
            sort_process.wait()

        closest_process = subprocess.Popen(
            ['bedtools','closest',
             '-a',temp_file.name,
             '-b',temp_file.name,
             '-io','-d',
            ],
            stdout = subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        cut_process = subprocess.Popen(
            ['cut','-f1-3,7'],
            stdin = closest_process.stdout,
            stdout = subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        distance_cutoff = get_distance_cutoff(temp_file.name)

        for line in cut_process.stdout:
            
            chrom, start, end, distance = line.strip().split('\t')
            distance = int(distance)

            if distance < 0:
                distance = 10000000

            print(
                chrom, start, end, log10(distance), distance < distance_cutoff,
                sep='\t',
                file=output
            )
