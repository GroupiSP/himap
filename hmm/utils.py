import numpy as np
import pickle
import json
import pandas as pd
from numba import jit
import os

#todo: documentation for all of the functions

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

def get_rmse(mean_rul_dict, true_rul_dict):
    df_results = pd.DataFrame(columns=['Name', 'rmse'])
    for key in mean_rul_dict.keys():
        predicted_values = mean_rul_dict[key]
        true_rul = true_rul_dict[key]
        true_values = list(range(true_rul, -1, -1))
        # Pad with zeros to ensure both arrays are the same length
        max_length = max(len(true_values), len(predicted_values))
        true_values = np.pad(true_values, (0, max_length - len(true_values)), constant_values=0)
        predicted_values = np.pad(predicted_values, (0, max_length - len(predicted_values)), constant_values=0)

        # Calculate RMSE
        rmse_pred = np.sqrt(np.mean((predicted_values - true_values)**2))
        new_row = pd.DataFrame([{'Name':key, 'rmse':rmse_pred}])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
    
    # Calculate and append the average coverage
    average_rmse = df_results['rmse'].mean()
    new_row = pd.DataFrame([{'Name':'Average', 'rmse':average_rmse}])
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    return df_results

def get_coverage(upper_bound_dict, lower_bound_dict, true_rul_dict):
    df_results = pd.DataFrame(columns=['Name', 'coverage'])
    for key in upper_bound_dict.keys():
        upper_bounds = upper_bound_dict[key]
        lower_bounds = lower_bound_dict[key]
        true_values = list(range(true_rul_dict[key], -1, -1))
        # Count the number of true values within the bounds
        count_within_bounds = sum(
            l <= t <= u for t, l, u in zip(true_values, lower_bounds, upper_bounds)
        )
        cov = count_within_bounds / len(true_values)
        new_row = pd.DataFrame([{'Name':key,'coverage':cov}])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
    # Calculate and append the average coverage
    average_coverage = df_results['coverage'].mean()
    new_row = pd.DataFrame([{'Name':'Average','coverage':average_coverage}])
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    return df_results

def calculate_area_weighted_by_time(x_values, y_values):
    area = 0
    for i in range(1, len(x_values)):
        interval = x_values[i] - x_values[0]
        area += interval * (y_values[i] + y_values[i-1]) / 2
    return area

def get_wsu(upper_bound_dict, lower_bound_dict):
    df_results = pd.DataFrame(columns=['Name', 'wsu'])
    for key in upper_bound_dict.keys():
        upper_bounds = upper_bound_dict[key]
        lower_bounds = lower_bound_dict[key]
        area_upper =  calculate_area_weighted_by_time(range(len(upper_bounds)), upper_bounds)
        area_lower = calculate_area_weighted_by_time(range(len(lower_bounds)), lower_bounds)
        area_wsu = area_upper - area_lower
        new_row = pd.DataFrame([{'Name':key,'wsu':area_wsu}])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
    
    # Calculate and append the average coverage
    average_wsu = df_results['wsu'].mean()
    new_row = pd.DataFrame([{'Name':'Average','wsu':average_wsu}])
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    return df_results

def evaluate_test_set(mean_rul_dict, upper_bound_dict, lower_bound_dict, true_rul_dict):
    df_rmse = get_rmse(mean_rul_dict, true_rul_dict)
    df_coverage = get_coverage(upper_bound_dict, lower_bound_dict, true_rul_dict)
    df_wsu = get_wsu(upper_bound_dict, lower_bound_dict)
    # Merge the dataframes on the 'name' column
    combined_df = pd.merge(df_rmse, df_coverage, on='Name')
    combined_df = pd.merge(combined_df, df_wsu, on='Name')
    return combined_df   

        


@jit(nopython=True)
def baumwelch_method(n_states, n_obs_symbols, logPseq, fs, bs, scale, score, history, tr, emi, calc_tr, calc_emi):
    score += logPseq
    logf = np.log(fs)
    logb = np.log(bs)
    logGE = np.log(calc_emi)
    logGTR = np.log(calc_tr)

    for i in range(n_states):
        for j in range(n_states):
            for h in range(len(history) - 1):
                scale_h1 = scale[0, h + 1]  # Pre-fetching to avoid complex indexing
                tr[i, j] += np.exp(logf[i, h] + logGTR[i, j] + logGE[j, history[h + 1] - 1] + logb[j, h + 1]) / scale_h1

    for i in range(n_states):
        for j in range(n_obs_symbols):
            # Create an empty list for indices where history == j + 1
            pos_indices = []
            for idx in range(len(history)):
                if history[idx] == j + 1:
                    pos_indices.append(idx)

            # Manually sum up values at the positions in pos_indices
            for pos in pos_indices:
                emi[i, j] += np.exp(logf[i, pos] + logb[i, pos])

    return tr, emi


@jit(nopython=True)
def fs_calculation(n_states, end_traj, fs, s, history, calc_emi, calc_tr):
    for count in range(1, end_traj):
        for state in range(n_states):
            fs[state, count] = calc_emi[state, history[count] - 1] * np.sum(fs[:, count - 1] * calc_tr[:, state])
        # scale factor normalizes sum(fs,count) to be 1.
        s[0, count] = np.sum(fs[:, count])
        fs[:, count] = fs[:, count] / s[0, count]
    return fs, s


@jit(nopython=True)
def bs_calculation(n_states, end_traj, bs, s, history, calc_emi, calc_tr):
    for count in range(end_traj - 2, -1, -1):
        for state in range(n_states):
            bs[state, count] = (1 / s[0, count + 1]) * np.sum(
                calc_tr[state, :].T * bs[:, count + 1] * calc_emi[:, history[count + 1] - 1])
    return bs


def calculate_expected_value(pmf_values):
    expected_value = sum(x * p for x, p in enumerate(pmf_values))
    return expected_value


def calculate_cdf(pmf, confidence_level):
    # Calculate the CDF
    cdf = np.cumsum(pmf)
    # Calculate the lower and upper percentiles
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    lower_value = np.argmax(cdf >= lower_percentile)
    upper_value = np.argmax(cdf >= upper_percentile)

    return lower_value, upper_value


def create_folders():
    """
    Create folders and subfolders for results
    :return: None
    """

    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")
        else:
            print(f"Folder already exists: {path}")

    # Create folders
    folder_path = os.path.join(os.getcwd(), "results")
    create_folder(folder_path)

    subfolder_names = ["dictionaries", "figures", "models"]

    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(os.getcwd(), "results", subfolder_name)
        create_folder(subfolder_path)
