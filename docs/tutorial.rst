
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

To start, you need to have the `locusregression` package and `bedtools` installed in a conda environemnt. You can check this
quickly by running:

.. code-block:: bash

    $ locusregression && bedtools
    
Next, for data you will need:

* VCF files with SBS mutations
* Any number of -or combination of- genomic correlates for your system
* A fasta file of your organism's genome (e.g. h19.fa)
* A chromsizes file of your organism's genome (e.g. hg19.chrom.sizes)

Fasta and chromsizes annotations can be found at the `UCSC data repository <https://genome.ucsc.edu/cgi-bin/hgGateway>_`


1. Data preparation
-------------------

**Regions**

First, we need to define some genomic regions which will serve as our "windows", or a segment of the genome which we
consider a locus. There are many ways one could define these regions, and simply dividing the genome into 
high-resolution 10-kilobase bins as I do is but one option.

To make this tutorial run faster, we will only model mutations on the chr1 of 96 esophogeal adenocarcino samples from
the PCAWG dataset, so we define a new genome with only the first chromosome:

.. code-block:: bash
    
    $ mkdir -p tutorial
    $ head -n1 hg19.chrom.sizes > tutorial/genome.txt

When modeling the full genome, it is a good idea to define a genome with only main chromosomes (chr1-N), removing alt scaffolds, etc.

Next, we can make our windows using the convenient command from bedtools:

.. code-block:: bash

    $ bedtools makewindows -g tutorial/genome.txt -w 10000 > tutorial/regions.bed

.. note::
   
   It is worthwhile to check that your windows are in sorted order, or you'll run into
   problems down the line:

   .. code-block:: bash

        $ sort -k1,1 -k2,2n --check tutorial/regions.bed

**Correlates**

Next, we need to associate each of our windows with values for some interesting genomic correlates. For example, if I have 1-kilobase
binned data for expression, replication timing, and H3K27ac in bed files, I can use:

.. code-block:: bash

    $ cat expression.bed | grep -v "NA" | sort -k1,1 -k2,2n | \
        bedtools map -a tutorial/regions.bed -b - -o sum -sorted -null 0.0 | \
        cut -f4 >> tutorial/expression.tsv

to aggregate and map that track data to our windows. We can do the same for the other correlates:

.. code-block:: bash

    $ cat replication_timing.bed | grep -v "NA" | sort -k1,1 -k2,2n | \
        bedtools map -a tutorial/regions.bed -b - -o sum -sorted -null 0.0 | \
        cut -f4 >> tutorial/replication_timing.tsv

    $ cat h3k27ac.bed | grep -v "NA" | sort -k1,1 -k2,2n | \
        bedtools map -a tutorial/regions.bed -b - -o sum -sorted -null 0.0 | \
        cut -f4 >> tutorial/h3k27ac.tsv

.. note::

    Again, it is very important to keep these data sorted and normalized. Above, 
    I include the `-sorted` flag in `bedtools map` to ensure ensure this. Also, I
    set `-null 0.0` so that winows which are not included in the track are still
    assigned a numerical value.
    

Finally, we can merge all of these correlates into one file:

.. code-block:: bash

    $ echo -e "#expression\t#replication_timing\t#h3k27ac" > tutorial/correlates.tsv
    $ paste tutorial/expression.tsv tutorial/replication_timing.tsv tutorial/h3k27ac.tsv >> tutorial/correlates.tsv

First, I added a commented header line to help the LocusRegression model keep track of what features 
are being used. Then, I just pasted together the files for each correlate.

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

.. note:: 

    Here, I model genomes from esophogeal cells, which I may assume all have similar genomic features/expression/etc. 
    Thus, I use only one "correlates" file which speeds up model calculation. If you wish to model a heterogeneous 
    collection of cells -biologically or technically- you can provide a sample-specific correlate and exposure file
    for each VCF file of mutations.


**Compiling a corpus**

With all of our data gathered and munged, we can compile a "corpus": a normalized and reformatted view of 
the data which is read by the LocusRegression model. For a list of VCF files stored in vcfs.txt:

.. code-block:: bash

    $ locusregression make-corpus \
        -vcf `cat vcfs.txt` \
        -fa hg19.fa \
        --genome tutorial/genome.txt \
        --regions-file tutorial/regions.bed \
        --correlates-file tutorial/correlates.tsv \
        -o tutorial/corpus.pkl

This will save the corpus to `tutorial/corpus.pkl`.


2. How many processes?
----------------------

Choosing the number of mixture components to describe a process is a perenial problem in topic modeling,
LocusRegression notwithstanding. Here, I employ random search of the model hyperparameter space paired
with a Successive Halving bandit to find the number of components which produces a descriptive but 
generalizeable model. This process can be parallelized for faster tuning.

To run the `tune` command, you have to give the path to corpus, as well as the minimum and maximum
bounds on the number of components to try. This command outputs a *tsv* file of scores for different
model configurations.

Additionally, I provided the `--tune-pi-prior` flag, which tells the tuner to try different values
for the dirichlet over mixtures for each sample, the I set `--seed-reps` to 3, which tells the tuner
to try each model configuration three times.

.. code-block:: bash

    $ locusregression tune \    
        --corpus tutorial/corpus.pkl \
        -o tutorial/grid.tsv \
        -min 3 -max 25 \
        --tune-pi-prior \
        --seed-reps 3 \
        --n-jobs 5

