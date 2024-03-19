#!/usr/bin/env python3

import subprocess
import tempfile
import logging
from .make_windows import check_regions_file
from numpy import array
import numpy as np
logger = logging.getLogger('Roadmap downloader')
logger.setLevel(logging.INFO)


def make_continous_features(*,
                bigwig_file, 
                regions_file,
                extend=None,
            ):

    check_regions_file(regions_file)

    with tempfile.NamedTemporaryFile() as bed:

        subprocess.check_output(
                        ['bigWigAverageOverBed',
                         f'-sampleAroundCenter={extend}' if extend is not None else '',
                         bigwig_file, 
                         regions_file, 
                         bed.name
                        ],
                    )

        sort_process = subprocess.Popen(
            ['sort', '-k1,1n', bed.name],
            stdout = subprocess.PIPE,
        )

        cut_process = subprocess.Popen(
            ['cut','-f5'],
            stdin = sort_process.stdout,
            stdout = subprocess.PIPE,
        )
        vals, err = cut_process.communicate()

        if not cut_process.returncode == 0:
            raise RuntimeError('Command returned non-zero exit status: ' + err.decode())
        
        vals = array(list(map(
            lambda x : float(x.strip()),
            vals.decode().strip().split('\n')
        )))

        return vals
    

def make_continous_features_bedgraph(*,
                bedgraph_file,
                regions_file,
                genome_file,
                extend=None,
                null = 'nan',
            ):
    
    check_regions_file(regions_file)

    if extend is not None:

        center_process = subprocess.Popen(
            ['awk','-v','OFS=\t',
            '{ center=$2+($3-$2)/2; print $1,center,center+1,$4 }', 
            regions_file],
            stdout = subprocess.PIPE,
        )

        slop_process = subprocess.Popen(
            ['bedtools','slop','-i','-','-g',genome_file,'-b', str(extend)],
            stdin=center_process.stdout,
            stdout=subprocess.PIPE,
        )

        input_process = subprocess.Popen(
            ['sort', '-k1,1', '-k2,2n'],
            stdin = slop_process.stdout,
            stdout = subprocess.PIPE,
        )
    else:
        input_process = subprocess.Popen(
            ['sort', '-k1,1', '-k2,2n', bedgraph_file],
            stdout = subprocess.PIPE,
        )

    map_process = subprocess.Popen(
                    ['bedtools','map',
                        '-a', '-',
                        '-b', bedgraph_file,
                        '-c', '4',
                        '-o', 'mean',
                        '-null', null,
                    ],
                    stdin=input_process.stdout,
                    stdout=subprocess.PIPE,
                )

    resort_process = subprocess.Popen(
        ['sort','-k4,4n'],
        stdin = map_process.stdout,
        stdout = subprocess.PIPE
    )

    cut_output = subprocess.check_output(
        ['cut','-f','5'],
        stdin=resort_process.stdout
    )

    vals = array(list(map(
            lambda x : float(x.strip()),
            cut_output.decode().strip().split('\n')
        )))

    assert len(vals) > 1

    return vals



def make_distance_features(
                        reverse=False,*,
                        genomic_features,
                        regions_file,
                    ):
    
    check_regions_file(regions_file)

    def _find_stranded_closest_feature(strand):

        strand_process = subprocess.Popen(
            ['awk','-v','OFS=\t',f'{{print $1,$2,$3,NR-1,0,"{strand}"}}', regions_file],
            stdout = subprocess.PIPE
        )

        closest_out = subprocess.check_output(
            ['bedtools','closest',
             '-a','-',
             '-b', genomic_features, 
             '-d',
             '-id',
             '-D', 'a','-t','first',
            ],
            stdin = strand_process.stdout,
        )

        strand_process.wait()

        return -array(
                list(map(
                    lambda x : x.split('\t')[-1], 
                    closest_out.decode().strip().split('\n')
                    ))
                ).astype(float)
    
    upstream = _find_stranded_closest_feature('+')
    downstream = _find_stranded_closest_feature('-')

    nan_mask = (upstream < 0.) | (downstream < 0.) | (upstream + downstream <= 0.)

    progress = upstream/(upstream + downstream + 1) 
    progress = np.minimum(progress, 1-progress)
    
    #progress = 1. - progress if reverse else progress

    total_distance = upstream + downstream

    progress[nan_mask] = np.nan
    total_distance[nan_mask] = np.nan
    
    return progress, total_distance



def make_discrete_features(
        column=4,
        null='none',
        class_priority = None,*,
        regions_file, 
        genomic_features,
    ):

    check_regions_file(regions_file)
    
    def _resolve_class_priority(vals):

        vals = set(vals).difference({null})

        if len(vals) == 0:
            return null
        elif len(vals) == 1:
            return vals.pop()
        else:
            for _class in class_priority:
                if _class in vals:
                    return _class
            else:
                raise RuntimeError(f'Could not resolve class priority for {vals} using {class_priority}')
    
    # check that the bedfile has 4 columns
    with open(genomic_features, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            cols = line.strip().split('\t')
            if len(cols) < column:
                raise ValueError(
                    f'Bedfile {genomic_features} must have at least {column} columns.'
                    'The fourth column should be the name of the class for that region.'
                )
            break

    map_out = subprocess.check_output(
        ['bedtools','map',
         '-a',regions_file,
         '-b', genomic_features,
         '-o','distinct',
         '-c',str(column),
         '-null', str(null),
         '-delim','|',
         '-split',
        ],
    )

    mappings = list(map(lambda x : x.split('\t')[-1], map_out.decode().strip().split('\n')))

    vals = [m.split('|') for m in mappings]
    classes = set([_v for v in vals for _v in v]).difference({null})

    if class_priority is None:
        class_priority = list(classes)
    else:
        assert set(class_priority) == classes, \
            f'Class priority must contain all classes in {classes}, non including the null class: {null}'
        
    vals = array([_resolve_class_priority(v) for v in vals])

    return vals

