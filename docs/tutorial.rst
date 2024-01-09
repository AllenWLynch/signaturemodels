
LocusRegression Tutorial
************************

LocusRegression is a statistical model which explains the observed mutations in a dataset of genomes, 
where each genome is described by a mixture of different processes which generated its mutations.
Processes, in turn, have a characteristic bias with respect to which genomic loci and nucleotides it affects. 
We model the locus bias of different processes with respect to their association with known genomic correlates 
like gene expression, acetylation, replication timing, etc. For each process, its mutational signature and association with
genomic correlates are inferred jointly from the data using a variational EM estimation method.

In this tutorial, I will explain how to:

1. Prepare and compile a dataset for modeling
2. Find the number of signatures which describes mutational processes in your data
3. Infer parameters of the generative model
4. Analyze the results

To start, you need to have the *locusregression* package, *bedtools*, *bcftools*, and *bigWigAverageOverBed* installed in a conda environemnt. You can check this quickly by running:

.. code-block:: bash

    $ locusregression && bedtools && bcftools && bigWigAverageOverBed

If one of these is not installed:

.. code-block:: bash

    $ conda install -c conda-forge -c bioconda -y bedtools bcftools ucsc-bigwigaverageoverbed 
    
Next, for data you will need:

* VCF files with SBS mutations
* Any number of -or combination of- genomic correlates for your system
* A fasta file of your organism's genome (e.g. h19.fa)
* A chromsizes file of your organism's genome (e.g. hg19.chrom.sizes)


1. Data preparation
-------------------

**Regions**

First, we need to define some genomic regions which will serve as our "windows", or a segment of the genome which we
consider a locus. There are many ways one could define these regions, and simply dividing the genome into 
high-resolution 10-kilobase bins as I do is but one option.

To start, I define a "genome" or "chrom sizes" file from a fasta:

.. code-block:: bash
    
    $ mkdir -p tutorial
    $ samtools faidx genomes/hg19.fa
    $ cut -f1-2 genomes/hg19.fa.fai | grep -vE "^chr[0-9Un]_" > tutorial/genome.txt

When modeling the full genome, it is a good idea to define a genome with only main chromosomes (chr1-N), removing alt scaffolds, etc.

Next, bin the genome using `get-regions`:

.. code-block:: bash

    $ locusregression get-regions \
        -g tutorial/genome.txt \
        -w 10000 \
        -v blacklist.bed \
        > tutorial/regions.bed

.. code-block:: bash

    $ locusregression trinucs \
        -r tutorial/regions.bed \
        -fa genomes/hg19.fa \
        -o tutorial/trinucs.npz


**Correlates**

Next, we need to associate each of our windows with values for some interesting genomic correlates. Currently, only
continous features from bigwig files are supported. For some bigwig with H3K27ac marks, use `ingest-bigwig` to
get the average value of the marks in each window. You must also provide a unique name:

.. code-block:: bash

    $ locusregression ingest-bigwig \
        H3K27ac.bigwig \
        -r tutorial/regions.bed \
        -name H3K27ac \
        -o tutorial/H3K27ac.txt

Check the output of this method to see the output format:

.. code-block:: bash

    $ head tutorial/H3K27ac.txt
    #H3K27ac
    0
    0.2577
    0.209125
    0.20075

These features are unnormalized, so you can add a script which normalizes them however you like.
For continous features, the log-transformation followed by standardization is reasonable.

For the next step, one must assemble a matrix of these features as a tsv file. After ingesting
any number of tracks, you can put together a combination of features into one tsv file using the `paste` command:

.. code-block:: bash

    $ paste tutorial/H3K27ac.txt tutorial/H3K36me3.txt > tutorial/correlates.tsv

**The locusregression software will not adjust the features you provide, so
be sure to normalize them beforehand.**


**Exposures**

The last data that we need to feed the model are "exposures" - which are technical
effects that could explain variation in the number of mutations we see for each window/locus. Supplying these
exposures allows the model to correct for their effects when modeling variable mutation rates across the genome.

A simple exposure one could provide is the read coverage within each window, which may be roughly proportional
to the ability to call a mutation at that locus. More sohpisticated models of sensitivity can also be used.

Provide exposures as a single column of positive values (a header is optional and is ignored):

.. code-block:: bash

    $ head -n3 exposures.txt
      0.01
      0.05
      0.45

The exposure file is the only optional input.


**Compiling a corpus**

A "Corpus" is a a normalized and reformatted view of the data which is read by the LocusRegression model, and
associates a set of mutations from multiple VCFs to some genomic correlates. The 
structure of your corpus also helps LocusRegression find the fastest method to perform parameter updates. 
Since we could assume samples from a certain cancer type have similar correlates, we can group all of the 
VCFs from a certain cancer type to type-specific correlates. If you wish to model multiple types together, 
just provide multiple corpuses to any of the methods below.

To produce a corpus for some hypothetical set of samples stored in `vcfs.txt`:

.. code-block:: bash

    $ locusregression corpus-make \
        -vcf `cat vcfs.txt` \
        -fa hg19.fa \
        --regions-file tutorial/regions.bed \
        --correlates-file tutorial/correlates.tsv \
        --trinuc tutorial/trinucs.npz \
        -o tutorial/corpus.h5 \
        --n-jobs 10 \
        --weight-col TCF \
        --chr-prefix chr # the VCF files only have numbers, but RefSeq has "chr1", for example

This will save the corpus to *tutorial/corpus.h5*.

**Note:** The `--weight-col` flag is optional, and allows you to specify an INFO column in the VCFs which contains
a weight for each mutation. This is useful if you want to weight mutations by their tumor cell fraction, for example.


2. How many processes?
----------------------

Choosing the number of components to describe a dataset is a perenial problem in topic modeling,
LocusRegression notwithstanding. Here, I employ random search of the model hyperparameter space paired
with a HyperBand bandit to find the number of components which produces a descriptive but 
generalizeable model. This process can be parallelized for faster tuning.

First, create a new "study", which will attempt to find the best hyperparameters for a certain model 
and data configuration:

.. code-block:: bash

    $ locusregression study-create \    
        --corpuses tutorial/corpus.h5 \
        -min 3 -max 12 \
        --study-name tutorial.1 \
        --fix-signatures SBS1 SBS2 SBS8 \
        --empirical-bayes \
        --model-type gbt \
        --num-epochs 200 

    [I 2023-10-29 16:12:11,918] A new study created in Journal with name: tutorial.1

The `--fix-signatures` flag is optional, and allows you to fix the signatures of certain processes to
known mutational signatures.

The `--empirical-bayes` flag is optional, and allows you to use empirical bayes to estimate the
prior distribution over signatures for each corpus supplied.

The `--model-type` flag is optional, and allows you to choose between a gradient-boosted tree model
(`gbt`) or a linear model (`regression`).


Now, by running the command:

.. code-block:: bash

    $ locusregression run-trial tutorial.1

and referencing the study name, a model is trained with a random set of hyperparameters and the result 
saved to the study. This process can be repeated as many times as desired, and can be parallelized.
I recommend running 100-200 trials to get a good sense of the hyperparameter space. Trials can be run
serially:

.. code-block:: bash

    $ for i in {1..100}; do locusregression run-trial tutorial.1 > $i.log 2>&1; done


or, in parallel while controlling the number of cores by having each process run a certain number of trials:

.. code-block:: bash

    $ for i in {1..5}; do locusregression run-trial tutorial.1 -i 40 > $i.log 2>&1 & done

The command above launches five processes in the background, each of which tries 40 model configurations.
Using a slurm server, one can simultaneously run numerous trials in different processes. I recommend
allocating 2500MB and 1 CPU per trial.

To get the results from the the tuning stage, run:

.. code-block::
    
    $ locusregression study-summarize tutorial.1 -o tutorial/tune_results.csv

From this CSV, you can manually choose the trial which produced the best model by eye-balling
the ELBO of the perplexity curve (**Lower is better**):

.. code-block:: bash

    $ locusregression retrain \
        tutorial.1 \
        --trial-num <best_trial> \
        -o tutorial/model.pkl

If you don't set `--trial-num`, the best trial will be chosen automatically using score only.


2b. How many processes? - Alternative
------------------------------------

If you already know how many processes are present in a sample, you can just do the following, and skip
step 3:

.. code-block:: bash

    $ locusregression train-model \
        -k 15 \
        -d tutorial/corpus.h5 \
        -o tutorial/model.pkl \
        --empirical-bayes \
        --fix-signatures SBS1 SBS2 SBS8 \
        --model-type gbt

You can test different values of `k` using a test set corpus:

.. code-block:: bash

    $ locusregression corpus-split tutorial/corpus.h5 \
        -to tutorial/train.h5 \
        -vo tutorial/test.h5

where `to` and `vo` stand for train out and validation out, respectively. Get the perplexity score
of the model on the validation corpus using:

.. code-block:: bash

    $ locusregression model-score \
        tutorial/model.pkl \
        -d tutorial/test.h5
 


3. Analysis
-----------

I am currently rebuilding the analysis CLI, but for now, three main methods are implemented. First,
`model-predict` produces the exposure matrix for each sample:

.. code-block:: bash

    $ locusregression model-predict \
        tutorial/model.pkl \
        -d tutorial/corpus.h5 \
        -o tutorial/exposures.csv

Next, `model-plot-summary` produces a plot of the signatures:

.. code-block:: bash

    $ locusregression model-plot-summary \
        tutorial/model.pkl \
        -o tutorial/summary.pdf


Finally, `model-mutation-rate-r2` evalutates the model's marginal mutation rate prediction against
the data form the provided corpus. The pseudo-R^2 score is reported (-1 to 1, higher is better):

.. code-block:: bash

    $ locusregression model-mutation-rate-r2 \
        tutorial/model.pkl \
        -d tutorial/corpus.h5


