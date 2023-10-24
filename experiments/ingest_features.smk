
rule all:
    input:
        #expand(config['datadir'] + '/{sample}_corpus.tsv', 
        #    sample = [k + ':' + v['roadmap_id'] for k,v in config['samples'].items()]
        expand(config['datadir'] + '/{sample}_features.normalized.tsv', 
               sample = [v['roadmap_id'] for k,v in config['samples'].items()]
        )
        


rule download_from_roadmap:
    output:
        temp(config['datadir'] + '/features/roadmap/bigwigs/{sample}-{signal}-{type}.bigwig')
    params:
        url = lambda w : config['roadmap']['url'].format(sample = w.sample, signal = w.signal, type = w.type)
    shell:
        'curl {params.url} -o {output}'


rule convert_to_bed:
    input:
        config['datadir'] + '/features/{source}/bigwigs/{sample}-{signal}-{type}.bigwig'
    output:
        temp(config['datadir'] + '/features/raw_bedfiles/{source}-{sample}-{signal}-{type}.bed')
    shell:
        'bigWigToBedGraph {input} {output}'



rule make_windows:
    input:
        genome = config['genome'],
        blacklist = config['blacklist']
    output:
        config['datadir'] + '/windows.bed'
    params:
        binsize = config['binsize']
    shell:
        'bedtools makewindows -g {input.genome} -w {params.binsize} | '
        'bedtools intersect -v -a - -b {input.blacklist} | sort -k1,1 -k2,2n > {output}'


def get_bedfile_name(w):
    if w.source == 'roadmap':
        return config['datadir'] + '/features/raw_bedfiles/{source}-{sample}-{signal}-{type}.bed'
    elif w.source == 'user_supplied':
        return config['user_supplied']['features'][w.signal]


rule map_to_windows:
    input:
        bedfile = get_bedfile_name,
        windows = rules.make_windows.output[0]
    output:
        temp(config['datadir'] + '/features/mapped_bedfiles/{source}-{sample}-{signal}-{type}.bed')
    shell:
        'cat {input.bedfile} | awk \'OFS="\t" {{print $1, $2, $3, "-", $4 }}\' | grep -v "NA" | '
        'bedtools map -a {input.windows} -b - -o sum -sorted -null 0.0 | '
        'cut -f4 > {output}'



def iterate_roadmap_config():

    for type, signals in config['roadmap']['features'].items():
            for signal in signals:
                yield (signal, type)


def aggregate_inputs(w):

    return [
            rules.map_to_windows.output[0].format(
                source = 'roadmap', sample = w.sample, 
                signal = signal, type = type
            )
            for signal, type in iterate_roadmap_config()
        ]
        
'''
[
    rules.map_to_windows.output[0].format(
        source = 'user', sample = w.sample, 
        signal = signal, type = 'user'
    )
    for signal in config['user_supplied']['features'].keys()
]
'''



def get_headers():
    return '\t'.join(['#' + signal for signal, type in iterate_roadmap_config()])


rule aggregate:
    input: aggregate_inputs
    output:
       temp(config['datadir'] + '/{sample}_features.tsv')
    params:
        headers = get_headers()
    shell:
        'echo "{params.headers}" > {output} && '
        'paste {input} >> {output}'


rule normalize_features:
    input: rules.aggregate.output
    output: temp(config['datadir'] + '/{sample}_features.normalized.tsv')
    params:
        normalize_script = config['normalization_script']
    shell: 
        'python {params.normalize_script} {input} {output}'


rule make_corpus:
    input: 
        features = rules.normalize_features.output,
        windows = rules.make_windows.output,
    params:
        vcfs = lambda w : ' '.join(config['samples'][w.sample]['vcfs']),
        fasta = config['fasta'],
        genome = config['genome'],
    output:
        config['datadir'] + '/{sample}_corpus.h5'
    shell:
        'locusregression make-corpus -vcfs {params.vcfs} '
        '-g {params.genome} -c {input.features} '
        '-r {input.windows} -fa {params.fasta} -o {output}'

