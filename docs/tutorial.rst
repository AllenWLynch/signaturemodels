
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

    $ locusregression

.. code-block:: bash
    
    $ bedtools


Next, for data you will need:

* VCF files with SBS mutations
* Any number of -or combination of- genomic correlates for your system
* A fasta file of your organism's genome (e.g. h19.fa)
* A chromsizes file of your organism's genome (e.g. hg19.chrom.sizes)

Fasta and chromsizes annotations can be found at the `UCSC data repository <https://genome.ucsc.edu/cgi-bin/hgGateway>_`


1. Data preparation
-------------------

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


