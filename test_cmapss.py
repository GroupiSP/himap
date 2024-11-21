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
hmm_c.fit(seqs_train,save_iters=False)
hmm_c.save_model()
hmm_c.prognostics(seqs_test, plot_rul=True)

# path_mean_rul = os.path.join(os.getcwd(), 'results', 'dictionaries', f"mean_rul_per_step_hsmm.json")
# path_pdf_rul = os.path.join(os.getcwd(), 'results', 'dictionaries', f"pdf_ruls_hsmm.json")
# path_upper_rul = os.path.join(os.getcwd(), 'results', 'dictionaries', f"upper_ruls_hsmm.json")
# path_lower_rul = os.path.join(os.getcwd(), 'results', 'dictionaries', f"lower_ruls_hsmm.json")
#
# with open(path_mean_rul, "r") as fp:
#     rul_mean_all=json.load( fp)
#
# with open(path_pdf_rul, "r") as fp:
#     pdf_ruls_all=json.load( fp)
#
# with open(path_upper_rul, "r") as fp:
#     rul_upper_bound_all=json.load( fp)
#
# with open(path_lower_rul, "r") as fp:
#     rul_lower_bound_all=json.load( fp)
