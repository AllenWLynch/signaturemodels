import os
import subprocess
import tempfile
import sys
from collections import defaultdict, Counter
import logging
logger = logging.getLogger('Windows')
logger.setLevel(logging.INFO)


def check_regions_file(regions_file):

    encountered_idx = defaultdict(lambda : False)

    with open(regions_file, 'r') as f:

        for i, line in enumerate(f):
            
            if line.startswith('#'):
                continue
            
            cols = line.strip().split('\t')
            assert len(cols) >= 12, \
                f'Expected 12 or more columns (in BED12 format) in {regions_file}, with the fourth column being an integer ID.\n'
            
            try:
                _, start, end, idx = cols[:4]
                idx = int(idx); start = int(start); end = int(end)
            except TypeError:
                raise TypeError(
                    f'Count not parse line {i} in {regions_file}: {line}.\n'
                    'Make sure the regions file is tab-delimited with columns chr<str>, start<int>, end<int>, idx<int>.\n'
                )
            
            encountered_idx[idx] = True

    assert len(encountered_idx) > 0, \
        f'Expected regions file to have at least one region.'

    largest_bin = max(encountered_idx.keys())
    assert all([encountered_idx[i] for i in range(largest_bin + 1)]), \
        f'Expected regions file to have a contiguous set of indices from 0 to {largest_bin}.\n'
    
    try:
        subprocess.check_output(['sort', '-k','1,1', '-k','2,2n', '-c', regions_file])
    except subprocess.CalledProcessError:
        raise ValueError(
            f'Expected regions file to be sorted by chromosome and start position.\n'
            f'Use \'sort -k1,1 -k2,2n -o <filename> <filename>\' to sort the file.\n'
        )


def _make_fixed_size_windows(*, 
                            genome_file, 
                            window_size,
                            blacklist_file=None,
                            output = sys.stdout
                        ):
    
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

    if blacklist_file is not None:
        subract_process = subprocess.Popen(
            ['bedtools', 'intersect', '-a', '-', '-b', blacklist_file, '-v'],
            stdin = sort_process.stdout,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )
        sort_process = subract_process

    add_id_process = subprocess.Popen(
        ['awk','-v','OFS=\t', '{print $0,NR-1,"0","+",$2,$3,"0,0,0","1",$3-$2,"0"}'],
        stdin = sort_process.stdout,
        stdout = output,
        **process_kw,
    )

    add_id_process.wait()


def _get_endpoints(allowed_chroms, *bedfiles):

    def _get_endpoints_bedfile(bedfile, track_id):
    
        with open(bedfile, 'r') as f:

            for line in f:

                if line.startswith('#'):
                    continue

                cols = line.strip().split('\t')

                if len(cols) < 3:
                    raise ValueError(f'Bedfile {bedfile} must have at least 3 columns')
                
                feature = '1' if len(cols) == 3 else cols[3]

                chrom, start, end = cols[:3]
                start = int(start); end = int(end)
                
                if chrom in allowed_chroms:
                    yield chrom, start, track_id, feature, True
                    yield chrom, end, track_id, feature, False

    endpoints = (
        endpoint
        for bedfile in bedfiles
        for endpoint in _get_endpoints_bedfile(bedfile, os.path.basename(bedfile))
    )
    
    return sorted(
        endpoints,
        key = lambda x : (x[0], x[1]),
    )


def _endpoints_to_regions(endpoints, min_windowsize = 0):

    active_features = Counter()
    feature_combination_ids = dict()
    prev_chrom = None; prev_pos = None

    for (chrom, pos, track_id, feature, is_start) in endpoints:

        pos = int(pos)

        if prev_chrom is None:
            prev_chrom = chrom; prev_pos = pos
        elif chrom != prev_chrom:
            active_features = Counter()
            prev_chrom = chrom; prev_pos = pos
        elif pos > (prev_pos + min_windowsize) and len(active_features) > 0:

            feature_combination = tuple(sorted(active_features.keys()))

            if not feature_combination in feature_combination_ids:
                feature_combination_ids[feature_combination] = len(feature_combination_ids)

            yield chrom, prev_pos, pos, feature_combination_ids[feature_combination]    

        if is_start:
            active_features[(track_id,feature)] += 1
        else:
            if active_features[(track_id,feature)] > 1:
                logger.warning(
                    f'Multiple overlapping features of the same type detected in file {track_id}, feature {feature} at position {chrom}:{pos}.\n'
                    'Make sure to merge the bedfile regions if this is intentional.'
                )
                active_features[(track_id,feature)]-=1
            else:
                active_features.pop((track_id,feature))

        prev_pos = pos; prev_chrom = chrom


def make_windows(
    *bedfiles,
    genome_file, 
    window_size, 
    blacklist_file, 
    output=sys.stdout, 
):
    
    logger.info(f'Checking sort order of bedfiles ...')
    for bedfile in bedfiles:
        subprocess.run(["sort", "-k1,1", "-k2,2n", "--check", bedfile], check=True)

    allowed_chroms=[]
    with open(genome_file,'r') as f:

        for line in f:
            if line.startswith('#'):
                continue
            allowed_chroms.append(line.strip().split('\t')[0].strip())

    logger.info(f'Using chromosomes: {", ".join(allowed_chroms)}')
    
    with tempfile.NamedTemporaryFile('w') as windows_file, \
        tempfile.NamedTemporaryFile('w') as pre_blacklist_file:

        _make_fixed_size_windows(
            genome_file=genome_file,
            window_size=window_size,
            output=windows_file,
        )
        windows_file.flush()
        
        for region in _endpoints_to_regions(
            _get_endpoints(
                allowed_chroms,
                windows_file.name,
                *bedfiles,
            )
        ):
            print(*region, sep='\t', file=pre_blacklist_file)
        pre_blacklist_file.flush()

        subract_process = subprocess.Popen(
            ['bedtools', 'intersect', '-a', pre_blacklist_file.name, '-b', blacklist_file, '-v'],
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=10000,
        )

        id_map = {}
        regions_collection = defaultdict(list)

        for line in subract_process.stdout:
            chr, start, end, window_id = line.strip().split('\t')
            window_id = id_map.setdefault(window_id, len(id_map))
            regions_collection[window_id].append((chr, int(start), int(end)))

        subract_process.wait()

        for window_id, regions in regions_collection.items():
            
            chrs, starts, ends = list(zip(*regions))
            
            region_start=min(starts); region_end=max(ends)
            region_chr = chrs[0]

            num_blocks = len(regions)
            
            block_sizes = ','.join(map(lambda x : str(x[0] - x[1]), zip(ends, starts)))
            block_starts = ','.join(map(lambda s : str(s - region_start), starts))

            print(region_chr,    # chr
                  region_start,  # start
                  region_end,    # end
                  window_id,     # name
                  '0','+',       # value, strand
                  region_start,  # thickStart
                  region_start,  # thickEnd
                  '0,0,0',       # itemRgb,
                  num_blocks,    # blockCount
                  block_sizes,   # blockSizes
                  block_starts,  # blockStarts
                  sep='\t', 
                  file=output
            )
