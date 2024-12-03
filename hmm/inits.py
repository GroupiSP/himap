import numpy as np
import scipy.stats

#todo: implement everything as methods of the classes and delete this file


def init_hmm_mc(hmm_class):
    tr = np.zeros((hmm_class.n_states, hmm_class.n_states))
    for i in range(hmm_class.n_states):
        if i < hmm_class.n_states - 1:
            tr[i, i] = np.random.uniform(0.95, 0.97)
            tr[i, i + 1] = 1 - tr[i, i]
        else:
            tr[i, i] = 1.0  # Make the last state absorbing
        hmm_class.tr = tr

    emi = np.zeros((hmm_class.n_states, hmm_class.n_obs_symbols))
    segment_size = hmm_class.n_obs_symbols // (hmm_class.n_states - 1)

    for i in range(hmm_class.n_states - 1):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        if i == hmm_class.n_states - 2:
            end_idx = hmm_class.n_obs_symbols - 1  # Leave room for the last observation
        obs_indices = np.arange(start_idx, end_idx)
        center = (start_idx + end_idx - 1) / 2
        gaussian = np.exp(-0.5 * ((obs_indices - center) ** 2) / ((end_idx - start_idx) / 4))
        emi[i, start_idx:end_idx] = gaussian
        emi[i, :] /= emi[i, :].sum()

        # Last row: all zeros except 1 at the end
    emi[-1, -1] = 1.0
    hmm_class.emi = emi
    
    
