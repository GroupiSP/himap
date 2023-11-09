import numpy as np
from inits import *
from utils import *
from base import GaussianHSMM

mc_sampling = True

if mc_sampling:
    hsmm_init = GaussianHSMM(n_states=6,
                             n_durations=260,
                             n_iter=100
                             )

    # command to initialize a model for MC sampling
    init4durval_observ_state_1(hsmm_init)

    # MC sampling command
    obs, states, means = MC_sampling(num_of_histories=10,
                                     timesteps=600,
                                     hsmm=hsmm_init
                                     )

    hsmm_estim = GaussianHSMM(n_states=6,
                              n_durations=140,
                              n_iter=10,
                              tol=0.5,
                              f_value=60,
                              obs_state_len=10,
                              left_to_right=True
                              )
    _, _, score = hsmm_estim.fit(obs)

    # fit with bic command
    # hsmm_estim, models, bic = hsmm_estim.fit_bic(obs, states=[2, 3, 4, 5, 6, 7, 8], return_models=True)

    mean_ruls, lower_ruls, upper_ruls = hsmm_estim.prognostics(obs,
                                                               max_timesteps=600
                                                               )
