from locusregression.corpus.sbs_corpus_maker import SBSCorpusMaker
import subprocess

def unstack_bed12_matrix(*,
    regions_file, 
    matrix,
    columns_names,
    output,
):
    pass

    regions = SBSCorpusMaker.read_windows(regions_file)
    assert matrix.shape[1] == len(regions), \
        f'Matrix has {matrix.shape[0]} rows, but {len(regions)} regions were found in the regions file.'
    
    segments = sorted(
        [
            (*segment, region.name)
            for region in regions
            for segment in region.segments()
        ],
        key=lambda x: (x[0], x[1])
    )

    print(*(['chrom', 'start', 'end'] + columns_names), sep='\t', file=output)

    for chrom, start, end, locus_idx in segments:
        print(
            chrom, start, end, *matrix[locus_idx], 
            sep='\t', 
            file=output
        )

            
