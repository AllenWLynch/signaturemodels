from .make_windows import check_regions_file
import numpy as np
from dataclasses import dataclass


@dataclass
class BED12Record:
    chromosome: str
    start: int
    end: int
    name: str
    score: int
    strand: str
    thick_start: int
    thick_end: int
    item_rgb: str
    block_count: int
    block_sizes: list[int]
    block_starts: list[int]

    def segments(self):
        for start, size in zip(self.block_starts, self.block_sizes):
            yield self.chromosome, self.start + start, self.start + start + size
        
    def __len__(self):
        return sum(self.block_sizes)
    
    def __str__(self):
        return '\t'.join(
            map(str, [self.chromosome, self.start, self.end, 
                      self.name, self.score, self.strand, 
                      self.thick_start, self.thick_end, 
                      self.item_rgb, self.block_count, 
                      ','.join(map(str, self.block_sizes)), 
                      ','.join(map(str, self.block_starts))
                    ]
            )
        )



def read_exposure_file(exposure_file, windows):
        
    exposures = []
    with open(exposure_file, 'r') as f:

        for lineno, txt in enumerate(f):
            if not txt[0] == '#':
                try:
                    exposures.append( float(txt.strip()) )
                except ValueError as err:
                    raise ValueError('Record {} on line {} could not be converted to float dtype.'.format(
                        txt, lineno,
                    )) from err

    assert len(exposures) == len(windows), 'The number of exposures provided in {} does not match the number of specified windows.\n'\
            'Each window must have correlates provided.'

    assert all([e >= 0. for e in exposures]), 'Exposures must be non-negative.'

    return np.array(exposures)[np.newaxis, :]



def read_windows(regions_file, sep = '\t'):

    def parse_bed12_line(line):
        chromosome, start, end, name, score, strand, thick_start, thick_end, item_rgb, block_count, block_sizes, block_starts = line.strip().split('\t')
        block_sizes = list(map(int, block_sizes.split(',')))
        block_starts = list(map(int, block_starts.split(',')))
        
        return BED12Record(
            chromosome=chromosome,
            start=int(start),
            end=int(end),
            name=name,
            score=float(score),
            strand=strand,
            thick_start=int(thick_start),
            thick_end=int(thick_end),
            item_rgb=item_rgb,
            block_count=int(block_count),
            block_sizes=block_sizes,
            block_starts=block_starts
        )

    check_regions_file(regions_file)
    
    windows = []
    with open(regions_file, 'r') as f:
        
        for lineno, txt in enumerate(f):
            if not txt[0] == '#':
                line = txt.strip().split(sep)
                assert len(line) >= 12, 'Expected BED12 file type with at least 12 columns'

            try:
                windows.append(
                    parse_bed12_line(txt)
                )
            except ValueError as err:
                raise ValueError('Could not ingest line {}: {}'.format(lineno, txt)) from err

    return windows

