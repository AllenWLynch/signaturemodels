import os
import subprocess
import tempfile
import sys
from collections import defaultdict


def _make_fixed_size_windows(*, 
                            genome_file, 
                            window_size,
                            blacklist_file=None,
                            output = sys.stdout):
    
    process_kw = dict(
        universal_newlines=True,
        bufsize=10000,
    )

    makewindows_process = subprocess.Popen(
        ['bedtools', 'makewindows', '-g', genome_file, '-w', str(window_size)],
        stdout = subprocess.PIPE,
        **process_kw,
    )

    sort_process = subprocess.Popen(
        ['sort', '-k1,1', '-k2,2n'],
        stdin = makewindows_process.stdout,
        stdout = subprocess.PIPE,
        **process_kw,
    )

    if sort_process is not None:
        subract_process = subprocess.Popen(
            ['bedtools', 'intersect', '-a', '-', '-b', blacklist_file, '-v'],
            stdin = sort_process.stdout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )
        sort_process = subract_process

    add_id_process = subprocess.Popen(
        ['awk', '{print $0"\t"NR-1}'],
        stdin = sort_process.stdout,
        stdout = output,
        **process_kw,
    )

    add_id_process.wait()


def _get_endpoints(*bedfiles, output = sys.stdout):

    def _get_endpoints_bedfile(bedfile, track_id):
    
        with open(bedfile, 'r') as f:

            for line in f:
                cols = line.strip().split('\t')

                if len(cols) < 3:
                    raise ValueError(f'Bedfile {bedfile} must have at least 3 columns')
                
                feature = '1' if len(cols) == 3 else cols[3]

                chrom, start, end = cols[:3]
                start = int(start); end = int(end)

                yield chrom, start, f'{track_id}:{feature};start'
                yield chrom, end, f'{track_id}:{feature};end'


    with tempfile.NamedTemporaryFile(mode = 'w') as temp:

        for bedfile in bedfiles:

            for _, row in enumerate(
                _get_endpoints_bedfile(bedfile, os.path.basename(bedfile))
            ):
                print(*row, sep = '\t', file = temp)
    
        temp.flush()

        sort_process = subprocess.Popen(
            ['sort', '-k1,1', '-k2,2n', temp.name],
            stdout = output,
            universal_newlines=True,
            bufsize=10000,
        )

        sort_process.wait()


def _endpoints_to_bed(endpoints_file, output = sys.stdout):

    active_features = set()
    feature_combination_ids = dict()
    prev_chrom = None; prev_pos = None

    with open(endpoints_file, 'r') as endpoints:

        for endpoint in endpoints:

            chrom, pos, endpoint_feature = endpoint.strip().split('\t')
            pos = int(pos)

            feature, endpoint_type = endpoint_feature.split(';')

            if prev_chrom is None:
                prev_chrom = chrom; prev_pos = pos
            elif chrom != prev_chrom:
                active_features = set()
                prev_chrom = chrom; prev_pos = pos
            elif pos != prev_pos and len(active_features) > 0:

                feature_combination = ';'.join(sorted(active_features))

                if not feature_combination in feature_combination_ids:
                    feature_combination_ids[feature_combination] = len(feature_combination_ids)
                    
                print(
                    chrom, prev_pos, pos, feature_combination_ids[feature_combination],
                    sep = '\t',
                    file = output,
                )

            if endpoint_type == 'start':
                active_features.add(feature)
            elif endpoint_type == 'end':
                active_features.remove(feature)

            prev_pos = pos; prev_chrom = chrom


def make_windows(
    *bedfiles,
    genome_file, 
    window_size, 
    blacklist_file, 
    output=sys.stdout, 
):
    
    with tempfile.NamedTemporaryFile('w') as endpoints_file,\
        tempfile.NamedTemporaryFile('w') as windows_file, \
        tempfile.NamedTemporaryFile('w') as pre_blacklist_file:

        _make_fixed_size_windows(
            genome_file=genome_file,
            window_size=window_size,
            output=windows_file,
        )
        windows_file.flush()

        _get_endpoints(
            windows_file.name,
            *bedfiles,
            output=endpoints_file,
        )
        endpoints_file.flush()

        _endpoints_to_bed(
            endpoints_file.name,
            output=pre_blacklist_file,
        )
        pre_blacklist_file.flush()

        subract_process = subprocess.Popen(
            ['bedtools', 'intersect', '-a', pre_blacklist_file.name, '-b', blacklist_file, '-v'],
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        id_map = {}
        for line in subract_process.stdout:
            chr, start, end, window_id = line.strip().split('\t')
            window_id = id_map.setdefault(window_id, len(id_map))
            print(chr, start, end, window_id, sep='\t', file=output)

        subract_process.wait()


def check_regions_file(regions_file):

    encountered_idx = defaultdict(lambda : False)
    with open(regions_file, 'r') as f:

        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            
            assert len( line.strip().split('\t') ) == 4, \
                f'Expected 4 columns in {regions_file}, with the fourth column being an integer ID.\n' \
                f'Add a column to the regions file using \'awk -v OFS="\\t" \'{{print $0,NR}}\' <filename>\''
            
            try:
                chr, start, end, idx = line.strip().split('\t')
                idx = int(idx); start = int(start); end = int(end)
            except ValueError:
                raise ValueError(
                    f'Count not parse line {i} in {regions_file}: {line}.\n'
                    'Make sure the regions file is tab-delimited with columns chr<str>, start<int>, end<int>, idx<int>.\n'
                )
            
            encountered_idx[idx] = True

    largest_bin = max(encountered_idx.keys())
    assert all([encountered_idx[i] for i in range(largest_bin + 1)]), \
        f'Expected regions file to have a contiguous set of indices from 0 to {largest_bin}.\n'

    
'''if __name__ == "__main__":

    with open('windows.bed','w') as f:
        
        make_windows(
            genome_file = '/Users/allenwlynch/genomes/hg19.chrom.sizes',
            window_size=10000000,
            blacklist_file='blacklist.bed',
            output=f,
            *['example.bed', 'example2.bed'],
        )

    check_regions_file('windows.bed')'''