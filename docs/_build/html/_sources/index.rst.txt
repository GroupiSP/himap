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
.. code-block:: bash

   pip install himap


.. toctree::
   :maxdepth: 1
   :caption: Documentation overview:

   quick_start
   fundamentals
   himap

Citing this repository:
=======================
*dummy citation*


Authors
=======
HiMAP was developed by the Intelligent Sustainable Prognostics (ISP) group at TU Delft, Aerospace Engineering Faculty.


- **Thanos Kontogiannis**: `a.kontogiannis@tudelft.nl <a.kontogiannis@tudelft.nl>`_
- **Mariana Salinas-Camus**: `m.salinascamus@tudelft.nl <m.salinascamus@tudelft.nl>`_
- **Nick Eleftheroglou**: `n.eleftheroglou@tudelft.nl <n.eleftheroglou@tudelft.nl>`_

.. |ISP| image:: _images/ISP.jpeg
   :height: 100px

.. |TU| image:: _images/TUDelft.png
   :height: 100px

|ISP| |TU| 