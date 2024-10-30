import numpy as np
import pickle
import json
import pandas as pd

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def create_data_hsmm(files, obs_state_len, f_value):
    """
    function to build degradation history array
    :param files: list of csv files of histories
    :param max_len: maximum length of histories
    :return: obs matrix ready for hsmm model
    """

    traj = {f"traj_{i}": list(pd.read_csv(files[i], usecols=[0])['clusters']) for i in range(len(files))}
    traj = fix_input_data(traj, f_value, obs_state_len)
    return traj


def MC_sampling(num, timesteps, HSMM):
    obs, states = {}, {}

    for i in range(num):
        sample = HSMM.sample(timesteps)
        _, obs1, states1 = sample

        for j in range(len(states1)):
            if states1[j] > states1[j + 1]:
                idx = j
                break
        obs.update({f'traj_{i + 1}': list(obs1[:idx + 1, 0])})
        states.update({f'traj_{i + 1}': list(states1[:idx + 1])})

    means = list(HSMM.mean[:, 0])
    return obs, states, means


# masks error when applying log(0)
def log_mask_zero(a):
    with np.errstate(divide="ignore", invalid='ignore'):
        return np.log(a)


def get_single_history(data, index):
    history = data[index, :].reshape((data.shape[1], 1))
    history = history[~np.all(history == 0, axis=1)]

    return history


def get_single_history_states(states, index, last_state):
    history_states = states[index]

    for j in range(len(history_states)):
        if history_states[j] == last_state:
            history_states = history_states[0:j + 1]
            break
    return history_states


def get_viterbi(HSMM, data):
    results = []
    keys = list(data.keys())
    for i in range(len(data)):
        history = np.array(data[keys[i]]).reshape((len(data[keys[i]]), 1))
        newstate_t = HSMM.predict(history)
        results.append(list(newstate_t[0]))

    return results


def fix_input_data(traj, f_value, obs_state_len, is_zero_indexed=True):
    """
    This function is used to fix the input data for the HSMM model. It appends the f_value to the end of each trajectory
    for the number of obs_state_len times. It also adds 1 to each value in the trajectory if is_zero_indexed is True.

    :param traj:
    :param f_value:
    :param obs_state_len:
    :param is_zero_indexed:
    :return:
    """
    assert isinstance(traj, dict), "Input data must be a dictionary"

    keys = list(traj.keys())
    if is_zero_indexed:
        for key in keys:
            traj[key] = [value + 1 for value in traj[key]]

    for key in keys:
        traj[key].extend([f_value for _ in range(obs_state_len)])
    return traj
