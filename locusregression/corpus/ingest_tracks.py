#!/usr/bin/env python3

import subprocess
import tempfile
import logging
from .make_windows import check_regions_file
import sys
from numpy import array
logger = logging.getLogger('Roadmap downloader')
logger.setLevel(logging.INFO)


def make_continous_features(*,
                bigwig_file, 
                regions_file,
            ):

    check_regions_file(regions_file)

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


def make_distance_features(*,
                        genomic_features,
                        regions_file,
                    ):
    
    check_regions_file(regions_file)

    def _find_stranded_closest_feature(strand):

        strand_process = subprocess.Popen(
            ['awk','-v','OFS=\t',f'{{print $0,0,"{strand}"}}', regions_file],
            stdout = subprocess.PIPE
        )

        closest_out = subprocess.check_output(
            ['bedtools','closest',
             '-a','-',
             '-b', genomic_features, 
             '-d',
             '-iu',
             '-D', 'a','-t','first',
            ],
            stdin = strand_process.stdout,
        )

        strand_process.wait()

        return list(map(
            lambda x : x.split('\t')[-1], 
            closest_out.decode().strip().split('\n')
        ))
    
    upstream = _find_stranded_closest_feature('+')
    downstream = _find_stranded_closest_feature('-')
    
    return array(upstream), array(downstream)



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
            

    map_out = subprocess.check_output(
        ['bedtools','map',
         '-a',regions_file,
         '-b',genomic_features,
         '-F', '0.5',
         '-o','distinct',
         '-c',str(column),
         '-null', str(null),
         '-delim','|'
        ],
    )

    mappings = list(map(lambda x : x.split('\t')[4], map_out.decode().strip().split('\n')))

    vals = [m.split('|') for m in mappings]
    classes = set([_v for v in vals for _v in v]).difference({null})

    if class_priority is None:
        class_priority = list(classes)
    else:
        assert set(class_priority) == classes, \
            f'Class priority must contain all classes in {classes}, non including the null class: {null}'
        
    vals = array([_resolve_class_priority(v) for v in vals])

    return vals

