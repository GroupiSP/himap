Quick Start Guide: Prognostics using C-MAPSS
*********************************************

This guide provides a step-by-step process for loading the C-MAPSS dataset, creating a Hidden Markov Model (HMM) and a Hidden Semi-Markov Model (HSMM), and performing prognostics using your model.

.. warning::
    This example will NOT run with the PyPi install, since the package does not 
    contain the example data. To get all of the required files, please build
    the entire GitHub repository, following the respective instructions.
        

.. contents:: Table of Contents
   :depth: 3
   :local:

Using the classic Hidden Markov Model (HMM) for prognostics
===========================================================

Step 1: Load the C-MAPSS data example
-------------------------------------

The data used corresponds to sensor 11 data from the FD001 sub-dataset of C-MAPSS, discretized into 20 values for easier use with HMMs. The training dataset consists of 80 run-to-failure degradation histories, while the testing dataset contains 20 run-to-failure degradation histories.

The dataset includes two files:

- ``train_FD001_disc_20_mod.csv`` for training
- ``test_FD001_disc_20_mod.csv`` for testing

To load the data, use the following Python code:

.. code-block:: python

   f_value = 21  # Failure value
   obs_state_len = 5  # Number of times the failure value is repeated in the data

   # Load training and testing data
   seqs_train, seqs_test = load_data_cmapss(f_value=f_value, obs_state_len=obs_state_len)

The ``f_value`` refers to the failure value (21), and ``obs_state_len`` indicates how many times this failure value is repeated in the training data. This ensures that the model learns a consistent failure pattern. To better understand the degradation process, visualize a few degradation histories from the training set. Below, you can see 8 example trajectories, showing how sensor readings change over time. Since this dataset comes from one failure mode (FD001), the degradation trends in training and testing are similar.  

.. image:: _images/train_quick_start.png
   :align: center
   :width: 600


Step 2: Create Your HMM
-----------------------

Next, define the Hidden Markov Model (HMM). The number of observation symbols is 21 due to the discretization of the data (20 values plus the failure value). The number of states can be adjusted depending on your preference.

.. code-block:: python

   hmm_c = HMM(n_obs_symbols=f_value)

You can also provide ``n_iter`` as an input, which specifies the maximum number of iterations for training the model.

Step 3: Optimize the Number of States and Train Your Model
----------------------------------------------------------

To determine the optimal number of states, you can use the Bayesian Information Criterion (BIC), which balances model complexity and fit:

.. code-block:: python

   hmm_c, bic = hmm_c.fit_bic(seqs_train, states=list(np.arange(2, 15)))

This tests models with 2 to 15 states (you can adjust the range) and returns the best one based on BIC.

If you already know the number of hidden states that you want to use, you can define them during the instantiation of the model with the ``n_states`` parameter and train your model directly:

.. code-block:: python
   
   hmm_c = HMM(n_states=5, n_obs_symbols=f_value)
   hmm_c.fit(seqs_train)

Step 4: Perform Prognostics
---------------------------

Once the model is trained, you can use the defined prognostic module to predict the Remaining Useful Life (RUL):

.. code-block:: python

   hmm_c.prognostics(seqs_test, plot_rul=True, get_metrics=True)

This function will generate and save RUL plots in a ``figures`` folder and also save a CSV file containing the following performance metrics:

- **RMSE**: Measures prediction accuracy (lower is better)
- **Coverage**: Indicates how well the true RUL values fall within the confidence intervals (ideal = 1)
- **WSU**: Represents the spread of uncertainty (higher values indicate wider confidence intervals)

Additionally, RUL probability distributions (PDFs) for each time step are saved in the ``dictionary`` folder, along with confidence intervals.

Below is an example of the RUL prediction results:

.. image:: _images/hmm_RUL_plot_traj_19.png
   :align: center
   :width: 600


Using a More Advanced Model: HSMM
=================================

The C-MAPSS data might be too complex for an HMM. To improve predictions, we can use a Hidden Semi-Markov Model (HSMM).

Step 1: Create Your HSMM
------------------------

An HSMM works similarly to an HMM, but states last for varying durations instead of transitioning at each step. Unlike HMMs, HSMMs don’t need predefined observation symbols. However, they require ``n_durations``, which is the maximum duration each state can have.

.. code-block:: python

   hsmm_c = GaussianHSMM(n_durations=200, f_value=f_value, obs_state_len=obs_state_len)

Step 2: Optimize the Number of States and Train Your Model
----------------------------------------------------------

Similar to the HMM, you can use the ``fit_bic`` method to optimize the number of states using the BIC criterion.

.. code-block:: python

   hsmm_c.fit_bic(seqs_train, states=list(np.arange(2, 7)))

Step 3: Perform Prognostics
---------------------------

With the trained HSMM, perform prognostics:

.. code-block:: python

   hsmm_c.prognostics(seqs_test, plot_rul=True, get_metrics=True)

By using HSMMs, you’ll likely see improved RUL predictions compared to HMMs! For the C-MAPSS, the RMSE improves, and the uncertainty confidence intervals reduce over time. 

.. image:: _images/hsmm_RUL_plot_traj_19.png
   :align: center
   :width: 600