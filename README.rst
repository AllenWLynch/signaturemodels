
LocusRegression
***************

A statistical model for jointly learning the genomic distribution and signatures of
mutational processes.

System requirements
-------------------

* Linux or Mac OS
* Python >= 3.6

Installation
------------

First, set up a conda environment in which we can install the model, then 
install the package using pip:

.. code-block :: bash

    (base) $ conda create --name locusregression -c conda-forge python>=3.6 pip seaborn
    (base) $ conda activate locusregression
    (locusregression) $ pip install git+https://github.com/AllenWLynch/signaturemodels.git@refactor

Dependencies
------------

* numpy
* scipy
* matplotlib
* pyfaidx


Tutorial
--------

Please see the `LocusRegression tutorial <docs/tutorial.rst>`_ for usage, and the `methods doc <docs/methods.pdf>`_ for information on the model.

For an overview of the simulation methods, see `simulation tutorial <docs/simulation_tutorial.ipynb>`_.
