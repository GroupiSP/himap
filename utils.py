import numpy as np
import pickle
import json


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def create_data_hsmm(files, max_len=2200):
    """
    function to build degradation history array
    :param files: list of csv files of histories
    :param max_len: maximum length of histories
    :return: obs matrix ready for hsmm model
    """

    obs_concat = np.zeros((len(files), max_len))

    for i, file in enumerate(files):
        obs = pd.read_csv(file, usecols=[0])
        obs1 = obs.to_numpy(copy=True).reshape((1, len(obs)))

        for j in range(len(obs)):
            if obs1[0, j] != 10:
                obs_concat[i, j] = (
                        obs1[0, j] + 1
                )  # HSMM requires the first observation to not be 0
                index = j

        # Creation of the last observed state (failure state) with final value=15 and duration=10
        # - assists in the model fit
        for k in range(10):
            obs_concat[i, index + 1 + k] = 15

    return obs_concat


def MC_sampling(num_of_histories, timesteps, hsmm, num2plot=False):
    obs = []
    states = []
    means = []

    for i in range(num_of_histories):
        sample = hsmm.sample(timesteps)
        _, obs1, states1 = sample

        for j in range(len(states1)):
            if states1[j] > states1[j + 1]:
                states1[j + 1:] = 0
                idx = j
                break
        obs1[idx + 1:] = 0
        obs.append(obs1)
        states.append(states1)

    obs_ar = np.asarray(obs)
    obs_ar = obs_ar.reshape((num_of_histories, timesteps))

    states_ar = np.asarray(states)
    states_ar = states_ar.reshape((num_of_histories, timesteps))

    means_ar = hsmm.mean


    return obs_ar, states_ar, means_ar


# masks error when applying log(0)
def log_mask_zero(a):
    with np.errstate(divide="ignore", invalid='ignore'):
        return np.log(a)


def save_results(HSMM, technique, score_per_sample=None, score_per_iter=None):
    with open(f'model_{technique}.txt', 'wb') as f:
        pickle.dump(HSMM, f)

    if score_per_sample is not None:
        with open(f'score_per_sample_{technique}.txt', 'wb') as f:
            pickle.dump(score_per_sample, f)

    if score_per_iter is not None:
        with open(f'score_per_iter_{technique}.txt', 'wb') as f:
            pickle.dump(score_per_iter, f)


def get_single_history(data, index):
    history = data[index, :].reshape((data.shape[1], 1))
    history = history[~np.all(history == 0, axis=1)]

    return history


def get_single_history_states(states, index, obs_state_len):
    history_states = states[index, :].reshape((states.shape[1], 1))

    for j in range(len(history_states)):
        if history_states[j, 0] > history_states[j + 1, 0]:
            history_states = history_states[0:j + 2 - obs_state_len, 0]
            break
    return history_states


def get_viterbi(HSMM, data):
    results = np.zeros_like(data)

    for i in range(data.shape[0]):
        history = get_single_history(data, i)
        newstate_t = HSMM.predict(history)
        pred_states = np.asarray(newstate_t[0])
        results[i, :len(pred_states)] = pred_states.copy()

    return np.asarray(results, dtype=int)


def bic(logliks, emission_matrix, transition_matrix, train):
    best_ll = max(logliks)
    n = len(train)
    dims_emission = emission_matrix.size
    hi_emission = (dims_emission(1) * dims_emission(2)) - dims_emission(1) - (dims_emission(2) + 1)
    dims_transition = transition_matrix.size
    hi_transition = 2 * (dims_transition(1) - 1)
    hi = hi_emission + hi_transition
    bic = best_ll - (hi / 2) * np.log(n)

    return bic
