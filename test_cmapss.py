# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:25:54 2024

@author: msalinascamus
"""

import pandas as pd
from base import *

train = pd.read_csv('train_FD001_disc_20_mod.csv', sep=';') #the discretized data starts from 0, to work with HMM it has to start from 1
test = pd.read_csv('test_FD001_disc_20_mod.csv', sep=';')
train_units = np.unique(train['unit_nr'].to_numpy())
test_units = np.unique(test['unit_nr'].to_numpy())
seqs_train = {}
for i, unit in enumerate(train_units):
    seq_unit = train.loc[train['unit_nr'] == unit]['s_discretized'].to_numpy() + 1
    failure = [21] * 5
    seq_unit = np.concatenate([seq_unit, failure]).tolist()
    seqs_train[f'traj_{i}'] = seq_unit

seqs_test = {}
for i, unit in enumerate(test_units):
    seq_unit = test.loc[test['unit_nr'] == unit]['s_discretized'].to_numpy() + 1
    failure = [21] * 5
    seq_unit = np.concatenate([seq_unit, failure]).tolist()
    seqs_test[f'traj_{i}'] = seq_unit

hmm_c = HMM(n_states=5, n_obs_symbols=21, left_to_right=True)
hmm_c.fit(seqs_train)
rul_mean_all, rul_upper_bound, rul_lower_bound = hmm_c.prognostics(seqs_test, plot_rul=True)