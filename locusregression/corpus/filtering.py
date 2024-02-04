import subprocess
import tempfile

def get_rainfall_statistic(vcf_file,*,output):

    def get_distance_cutoff(mutations_file):
        pass

    
    with tempfile.NamedTemporaryFile() as temp_file:

        filter_process = subprocess.Popen(
                ['bcftools','view','-v','snps', vcf_file],
                stdout = subprocess.PIPE,
                universal_newlines=True,
                bufsize=10000,
                stderr = subprocess.DEVNULL,
            )
        
        with open(tempfile,'w') as f:
            query_process = subprocess.Popen(
                ['bcftools','query','-f', '%CHROM\t%POS0\t%POS0' + '\n',],
                stdin = filter_process.stdout,
                stdout = f,
                universal_newlines=True,
                bufsize=10000,
            )
            query_process.wait()

        closest_process = subprocess.Popen(
            ['bedtools','closest',
             '-a',temp_file.name,
             '-b',temp_file.name,
             '-io',
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

            print(
                chrom, start, end, distance, distance < distance_cutoff,
                sep='\t',
                file=output
            )
