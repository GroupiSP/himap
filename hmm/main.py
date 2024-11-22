import numpy as np
from inits import *
from utils import *
from base import GaussianHSMM
from plot import plot_multiple_observ

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

mc_sampling = True

if mc_sampling:
    hsmm_init = GaussianHSMM(n_states=6,
                             n_durations=260,
                             n_iter=100
                             )

    # command to initialize a model for MC sampling
    init4durval_observ_state_1(hsmm_init)

    # MC sampling command
    num_of_histories = 20
    max_timesteps = 600
    obs, states, means = MC_sampling(num_of_histories, max_timesteps, hsmm_init)

    plot_multiple_observ(obs, states, num2plot=5)

    hsmm_estim = GaussianHSMM(n_states=6,
                              n_durations=140,
                              n_iter=2,
                              tol=0.5,
                              f_value=60,
                              obs_state_len=10,
                              left_to_right=True
                             )

    hsmm_estim.fit(obs, save_iters=True)

    hsmm_estim.save_model()

    # fit with bic command
    # hsmm_estim, models, bic = hsmm_estim.fit_bic(obs, states=[2, 3, 4, 5, 6, 7, 8], return_models=True)

    hsmm_estim.prognostics(obs, plot_rul=True)

