HiMAP: Hidden Markov models for Advanced Prognostics
====================================================

HiMAP is a Python package for implementing hidden Markov Models (HMMs) and hidden semi-Markov Models (HSMMs) tailored for prognostic applications. It provides a probabilistic framework for predicting Remaining Useful Life (RUL) and modeling complex degradation processes without requiring labeled datasets.

Key Features:
-------------
- HMM and HSMM implementations: Unsupervised stochastic models for system degradation.
- Core Methods: Includes methods for model training, state inference, and data generation through Monte Carlo sampling.
- Probabilistic RUL prediction: computes RUL as a probability density function (pdf) using Viterbi-decoded state sequences.

Installation:
=============

Option 1: Install via pip
-----------------------------
The easiest way to install HiMAP is through `pip`. Note that to install the package you need `Python 3.9+`. Simply run the following command: 

.. code-block:: bash

   pip install himap


Option 2: Install from source
-----------------------------
If you prefer to install HiMAP directly from the source, follow these steps:
1. Create a virtual environment and activate it. (This example will be demonstrated with Anaconda, but it is not required.)

- Step 1a: 

.. code-block:: bash

   conda create -n himap_env python=3.9 -y

- Step 1b: 

.. code-block:: bash

   conda activate himap_env

2. This repository can be directly pulled through GitHub by the following commands:

- Step 2a: 

.. code-block:: bash

   conda install git

- Step 2b:

.. code-block:: bash

   git clone https://github.com/GroupiSP/himap.git

- Step 2c:

.. code-block:: bash

   cd himap

3. The dependencies can be installed using the requirements.txt file

.. code-block:: bash

   pip install -r requirements.txt

4. To compile the Cython code, run the following command

.. code-block:: bash

   python setup_cython.py build_ext --inplace



.. toctree::
   :maxdepth: 1
   :caption: Documentation overview:

   quick_start
   fundamentals
   himap
   Github Repository <https://github.com/GroupiSP/himap>

Citing this repository:
=======================
If you use HiMAP in your research, please use the following citation:

.. code-block:: bibtex

   @software{kontogiannis_2026_18418216,
      author = {Kontogiannis, Thanos and Salinas-Camus, Mariana and Eleftheroglou, Nick},
      title  = {HiMAP: Hidden Markov models for Advanced Prognostics},
      month  = jan,
      year   = 2026,
      publisher = {Zenodo},
      version = {v1.3.0},
      doi = {10.5281/zenodo.18418216},
      url = {https://doi.org/10.5281/zenodo.18418216},
      swhid = {swh:1:dir:447c3c9e6743e4b56015f2107dbcd6d0eebe1bda
                   ;origin=https://doi.org/10.5281/zenodo.18418215;vi
                   sit=swh:1:snp:66b6fd9f335c7a71c1d805bc6cf19261589a
                   0770;anchor=swh:1:rel:3c6b8da54a15799b589065dbedc6
                   5540af805806;path=GroupiSP-himap-103efdc
                  },
   }


Authors
=======
HiMAP was developed by the Intelligent System Prognostics (ISP) group at TU Delft, Aerospace Engineering Faculty.


- **Thanos Kontogiannis**: `a.kontogiannis@tudelft.nl <a.kontogiannis@tudelft.nl>`_
- **Mariana Salinas-Camus**: `m.salinascamus@tudelft.nl <m.salinascamus@tudelft.nl>`_
- **Nick Eleftheroglou**: `n.eleftheroglou@tudelft.nl <n.eleftheroglou@tudelft.nl>`_

.. |ISP| image:: _images/ISP.jpeg
   :height: 100px

.. |TU| image:: _images/TUDelft.png
   :height: 100px

|ISP| |TU| 
