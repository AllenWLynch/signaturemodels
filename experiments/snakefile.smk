
rule all:
    input:
        [config['datadir'] + '/evaluations/{id}_eval.tsv'.format(id = _id) for _id in config['ids'].split(',') if not _id is '']

rule make_corpus:
    input: 
        config['configdir'] + '/{id}_config.pkl'
    output:
        corpus = config['datadir'] + '/simulations/{id}_corpus.h5',
        genparams = config['datadir'] + '/simulations/{id}_generative_params.pkl'
    params:
        prefix = lambda w : config['datadir'] + f'/simulations/{w.id}_'
    shell:
        "locusregression simulate --config {input} --prefix {params.prefix}"


rule tune_model:
    input:
        rules.make_corpus.output['corpus']
    output:
        config['datadir'] + '/tune_results/{id}_tuneresults.json'
    params:
        min_components = config['min_components'],
        max_components = config['max_components'],
    threads: 5
    resources:
        mem_mb = 2000
    shell:
        'locusregression tune -d {input} -o {output} '
        '--tune-subsample -min {params.min_components} -max {params.max_components} -j {threads}'


rule retrain:
    input:
        tune_results = rules.tune_model.output[0],
        corpus = rules.make_corpus.output['corpus']
    output:
        config['datadir'] + '/models/{id}_model.pkl'
    shell:
        'locusregression retrain -d {input.corpus} -r {input.tune_results} -o {output}'


rule evaluate:
    input:
        model = rules.retrain.output[0],
        genparams = rules.make_corpus.output['genparams']
    output:
        config['datadir'] + '/evaluations/{id}_eval.tsv'
    shell:
        'locusregression eval-sim -sim {input.genparams} --model {input.model} > {output}'

