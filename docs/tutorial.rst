
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

To start, you need to have the *locusregression* package and *bedtools* installed in a conda environemnt. You can check this
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

Another good idea is to remove regions of the genome which are hard to map or are known to caused biased signals. For instance, you could
remove ENCODE blacklist regions from your region set:

.. code-block:: bash

    $ bedtools intersect -v -a tutorial/regions.bed -b encode.blacklist.bed > tutorial/filtered_regions.bed \
        && mv tutorial/filtered_regions.bed tutorial/regions.bed

**Correlates**

Next, we need to associate each of our windows with values for some interesting genomic correlates. For example, if I have 1-kilobase
binned data for expression, replication timing, and H3K27ac in bed files, I can use:

.. code-block:: bash

    $ cat expression.bed | grep -v "NA" | sort -k1,1 -k2,2n | \
        bedtools map -a tutorial/regions.bed -b - -o sum -sorted -null 0.0 | \
        cut -f4 > tutorial/expression.tsv

to aggregate and map that track data to our windows. We can do the same for the other correlates:

.. code-block:: bash

    $ cat replication_timing.bed | grep -v "NA" | sort -k1,1 -k2,2n | \
        bedtools map -a tutorial/regions.bed -b - -o sum -sorted -null 0.0 | \
        cut -f4 > tutorial/replication_timing.tsv

    $ cat h3k27ac.bed | grep -v "NA" | sort -k1,1 -k2,2n | \
        bedtools map -a tutorial/regions.bed -b - -o sum -sorted -null 0.0 | \
        cut -f4 > tutorial/h3k27ac.tsv

..

    **Note:**
    Again, it is very important to keep these data sorted and normalized. Above, 
    I include the `-sorted` flag in `bedtools map` to ensure ensure this. Also, I
    set `-null 0.0` so that winows which are not included in the track are still
    assigned a numerical value.
    
**The *locusregression* software will not adjust the features you provide, so
be sure to standardize them beforehand.**

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

..

    **Note:**
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

This will save the corpus to *tutorial/corpus.pkl*.


1. How many processes?
----------------------

Choosing the number of mixture components to describe a process is a perenial problem in topic modeling,
LocusRegression notwithstanding. Here, I employ random search of the model hyperparameter space paired
with a Successive Halving bandit to find the number of components which produces a descriptive but 
generalizeable model. This process can be parallelized for faster tuning.

To run the *tune* command, you have to give the path to corpus, as well as the minimum and maximum
bounds on the number of components to try. This command outputs a *tsv* file of scores for different
model configurations.

.. code-block:: bash

    $ locusregression tune \    
        --corpus tutorial/corpus.pkl \
        -min 3 -max 12 \
        --n-jobs 5 \
        -o tutorial/grid.tsv \

We can plot the results in the *tutorial/grid.tsv* file to see which values for *n_components* make sense
for the dataset:

.. code-block::python

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    grid = pd.read_csv('data/tutorial/grid2.tsv', sep = '\t')

    sns.scatterplot(
        data = grid,
        x = 'param_n_components',
        y = 'mean_test_score',
        hue = 'iter',
        palette='coolwarm',
        s = 50,
        edgecolor = 'black',
        ax = ax,
    )
    sns.despine()
    ax.set(ylabel = 'Score', xlabel = 'N components')

.. image:: images/tuning.svg
    :width: 400

The SuccessiveHalving bandit runs "tournaments", where models are trained for a certain number of 
epochs, then tested. The best performing models are promoted to the next iteration and trained 
for more epochs. This process repeats until a group of winners is chosen.

Here, five or ten components gives a good fit for the dataset - I chose to use ten.

3. Training the model
---------------------

To train the representative model for the dataset, provide paths for the corpus and output, then
specify the chosen number of components using the "-k" argument:

.. code-block:: bash

    $ locusregression train-model \
        -d tutorial/corpus.pkl \
        -o tutorial/model.pkl \
        -k 10

4. Analysis
-----------

For this section, it is most natural to use an interactive tool like Jupyter notebooks to explore
the model and data. First, let's import some packages:

.. code-block:: python

    import locusregression
    import seaborn as sns
    import matplotlib.pylot as plt

The first thing we can do with a trained model is to see what signatures were uncovered and 
what genomic correlates they were associated with.

Load the model:

.. code-block:: python

    model = locusregression.load('tutorial/model.pkl')

Then, plot a signature like so:

.. code-block:: python

    model.plot_signature(1)

.. image:: docs/images/signature_example.svg
    :width: 400

And to see the signature's genomic correlate regression coefficients:

.. code-block:: python

    model.plot_coefficients(1)

.. image:: docs/images/coefs.svg
    :width: 400

This component is very anticorrelated with expressed genes, and looks something like
COSMIC signature SBS17b.

The locusregression model computes a posterior distribution for each
mutation which describes the probability that it was generated by each component/process. 
The model also calculates a mutation rate for each sample which is conditioned on the 
processes defining it.

We can compute and visualize these locus-based attributes of the data:

.. code-block:: bash

    corpus = locusregression.load_corpus('tutorial/corpus.pkl') # load corpus

    phi = model.get_phi_locus_distribution(corpus) # compute posterior over components for each mutation

    mutation_rate = model.get_expected_mutation_rate(corpus[2]) # get mutation rate for a sample

Now, we can plot. The top plot shows the probability that each mutation was generated by process 1. Next,
I plot the expression correlate. Last, I show the expected mutation rate across loci. The true loci
of the mutations are plotted as rug on the bottom plot. 

.. code-block:: bash

    fig, ax = plt.subplots(3,1,figsize=(20,4), sharex=True)
    
    sns.scatterplot(
        x = range(model.n_loci),
        y = phi[1], # plot first process
        s = 1,
        ax = ax[0],
        color = sns.color_palette("Set1")[0],
    )

    sns.scatterplot(
        x = range(model.n_loci),
        y = corpus[0]['X_matrix'][0,:],
        s = 1,
        ax = ax[1],
        color = sns.color_palette("Set1")[1],
    )


    sns.scatterplot(
        x = range(model.n_loci),
        y = mutation_rate,
        color = sns.color_palette("Set1")[2],
        s = 1,
        ax = ax[2],
    )

    sns.rugplot(
        x = corpus[0]['locus'],
        ax = ax[2],
        height=0.1,
        alpha = 0.1,
        color = 'black',
    )
    ax[0].set(ylabel = 'P(z=1 | m, l)')
    ax[1].set(ylabel = 'Expression')
    ax[2].set(ylabel = 'Mutation rate')
    sns.despine()

.. image:: docs/images/mutation_rate.svg
    :width: 800

Some areas of high mutational density are accounted for, but clearly more feature are needed to 
get a better fit.

Finally, to get the posterior distribution over processes for each sample, you can use:

.. code-block:: bash

    processes = model.predict(corpus)




