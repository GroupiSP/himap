# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors

import numpy as np
import pickle
import json
import pandas as pd
from numba import jit
import argparse
from importlib import resources
import os


class NumpyArrayEncoder(json.JSONEncoder):
    '''
    Custom JSON encoder to handle numpy.ndarray and numpy.integer objects for serialization.
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_data_hsmm(files, obs_state_len, f_value):
    """
    Creates a dictionary of trajectories for input into the HSMM model.

    Parameters
    ----------
    files : list of str
        List of file paths to CSV files containing trajectory data.
    obs_state_len : int
        The length of the observed state.
    f_value : float
        A value used for fixing the input data.

    Returns
    -------
    traj : dict
        A dictionary where keys are trajectory identifiers and values are lists of cluster data.
    """
    
    traj = {f"traj_{i}": list(pd.read_csv(files[i], usecols=[0])['clusters']) for i in range(len(files))}
    traj = fix_input_data(traj, f_value, obs_state_len)
    return traj


def load_data_cmapss(obs_state_len=5, f_value=21):
    """
    Loads the C-MAPSS dataset and prepares it for input into the HSMM model.

    Parameters
    ----------
    obs_state_len : int, optional
        Length to be used for the failure state, by default 5
    f_value : int, optional
        Failure value corresponding to the final state, by default 21

    Returns
    -------
    seqs_train: dict
        A dictionary containing the training trajectories.
    seqs_test: dict
        A dictionary containing the testing trajectories.
    """

    # Resolve packaged CSVs (works from any working directory once included in the wheel)
    train_res = resources.files("himap").joinpath("example_data", "train_FD001_disc_20_mod.csv")
    test_res = resources.files("himap").joinpath("example_data", "test_FD001_disc_20_mod.csv")

    with train_res.open("rb") as f:
        train = pd.read_csv(f, sep=";")

    with test_res.open("rb") as f:
        test = pd.read_csv(f, sep=";")

    train_units = np.unique(train['unit_nr'].to_numpy())
    test_units = np.unique(test['unit_nr'].to_numpy())
    seqs_train = {}
    for i, unit in enumerate(train_units):
        seq_unit = train.loc[train['unit_nr'] == unit]['s_discretized'].to_numpy() + 1
        failure = [f_value] * obs_state_len
        seq_unit = np.concatenate([seq_unit, failure]).tolist()
        seqs_train[f'traj_{i}'] = seq_unit

    seqs_test = {}
    for i, unit in enumerate(test_units):
        seq_unit = test.loc[test['unit_nr'] == unit]['s_discretized'].to_numpy() + 1
        failure = [f_value] * obs_state_len
        seq_unit = np.concatenate([seq_unit, failure]).tolist()
        seqs_test[f'traj_{i}'] = seq_unit

    return seqs_train, seqs_test

# masks error when applying log(0)
def log_mask_zero(a):
    """
    Applies the log function to an array, masking zero values.

    Parameters
    ----------
    a : np.ndarray
        An array of values.

    Returns
    -------
    np.ndarray
        The log-transformed array with zero values masked.
    """

    with np.errstate(divide="ignore", invalid='ignore'):
        return np.log(a)


def get_single_history_states(states, index, last_state):
    """
    Returns the history states for a single trajectory.

    Parameters
    ----------
    states : list
        A list of list, each list contains the states for a trajectory. 
    index : int
        The index of the trajectory
    last_state : int
        The last state of the trajectory

    Returns
    -------
    history_states : list
        A list of the history states for the trajectory.
    """

    history_states = states[index]

    for j in range(len(history_states)):
        if history_states[j] == last_state:
            history_states = history_states[0:j + 1]
            break
    return history_states


def get_viterbi(HSMM, data):
    """
    Applies the Viterbi algorithm to predict the most probable states for each trajectory in data using the HSMM.

    Parameters
    ----------
    HSMM : HSMM
        The trained Hidden Semi-Markov Model used to predict states.
    data : dict[str, List[int]]
        A dictionary of trajectories where each key is a trajectory name and each value is a list of observations.

    Returns
    -------
    results : List[List[int]]
        A list of lists containing the predicted states for each trajectory.
    """

    results = []
    keys = list(data.keys())
    for i in range(len(data)):
        history = np.array(data[keys[i]]).reshape((len(data[keys[i]]), 1))
        newstate_t = HSMM.predict(history)
        results.append(list(newstate_t[0]))

    return results


def fix_input_data(traj, f_value, obs_state_len, is_zero_indexed=True):
    """
    Prepares trajectory data for input into the HSMM model by appending f_value and adjusting indexing if needed.

    Parameters
    ----------
    traj : dict[str, List[int]]
        A dictionary containing the trajectories as lists of observed states.
    f_value : int
        The value to append to each trajectory.
    obs_state_len : int
        The number of times to append f_value to each trajectory.
    is_zero_indexed : bool, optional
        Flag indicating whether the data is zero-indexed. Default is True.

    Returns
    -------
    traj : dict[str, List[int]]
        The modified trajectory dictionary with f_value appended and indexing adjusted if necessary.
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
    """
    Computes the Root Mean Square Error (RMSE) between predicted Remaining Useful Life (RUL) and true RUL.

    Parameters
    ----------
    mean_rul_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of predicted RUL values.
    true_rul_dict : dict[str, int]
        A dictionary where each key is a trajectory name and the value is the true RUL for that trajectory.

    Returns
    -------
    df_results : pd.DataFrame
        A DataFrame containing RMSE values for each trajectory, including the average RMSE.
    """

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
        rmse_pred = np.sqrt(np.mean((predicted_values - true_values) ** 2))
        new_row = pd.DataFrame([{'Name': key, 'rmse': rmse_pred}])
        df_results = pd.concat([df_results, new_row], ignore_index=True)

    # Calculate and append the average coverage
    average_rmse = df_results['rmse'].mean()
    new_row = pd.DataFrame([{'Name': 'Average', 'rmse': average_rmse}])
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    return df_results


def get_coverage(upper_bound_dict, lower_bound_dict, true_rul_dict):
    """
    Calculates the coverage of true RUL values within the predicted upper and lower bounds.

    Parameters
    ----------
    upper_bound_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of upper bounds for predicted RUL.
    lower_bound_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of lower bounds for predicted RUL.
    true_rul_dict : dict[str, int]
        A dictionary where each key is a trajectory name and the value is the true RUL for that trajectory.

    Returns
    -------
    df_results : pd.DataFrame
        A DataFrame containing coverage values for each trajectory, including the average coverage.
    """
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
        new_row = pd.DataFrame([{'Name': key, 'coverage': cov}])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
    # Calculate and append the average coverage
    average_coverage = df_results['coverage'].mean()
    new_row = pd.DataFrame([{'Name': 'Average', 'coverage': average_coverage}])
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    return df_results


def calculate_area_weighted_by_time(x_values, y_values):
    """
    Calculates the area under the curve weighted by time for the given x and y values.

    Parameters
    ----------
    x_values : list[int]
        A list of x values (e.g., time).
    y_values : list[float]
        A list of y values (predicted values).

    Returns
    -------
    area : float
        The area under the curve weighted by time.
    """
    area = 0
    for i in range(1, len(x_values)):
        interval = x_values[i] - x_values[0]
        area += interval * (y_values[i] + y_values[i - 1]) / 2
    return area


def get_wsu(upper_bound_dict, lower_bound_dict):
    """
    Computes the Weighted Spread Uncertainty (WSU) between the upper and lower bounds.

    Parameters
    ----------
    upper_bound_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of upper bounds for predicted RUL.
    lower_bound_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of lower bounds for predicted RUL.

    Returns
    -------
    df_results : pd.DataFrame
        A DataFrame containing WSU values for each trajectory, including the average WSU.
    """
    df_results = pd.DataFrame(columns=['Name', 'wsu'])
    for key in upper_bound_dict.keys():
        upper_bounds = upper_bound_dict[key]
        lower_bounds = lower_bound_dict[key]
        area_upper = calculate_area_weighted_by_time(range(len(upper_bounds)), upper_bounds)
        area_lower = calculate_area_weighted_by_time(range(len(lower_bounds)), lower_bounds)
        area_wsu = area_upper - area_lower
        new_row = pd.DataFrame([{'Name': key, 'wsu': area_wsu}])
        df_results = pd.concat([df_results, new_row], ignore_index=True)

    # Calculate and append the average coverage
    average_wsu = df_results['wsu'].mean()
    new_row = pd.DataFrame([{'Name': 'Average', 'wsu': average_wsu}])
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    return df_results


def evaluate_test_set(mean_rul_dict, upper_bound_dict, lower_bound_dict, true_rul_dict):
    """
    Evaluates the test set by calculating RMSE, coverage, and WSU.

    Parameters
    ----------
    mean_rul_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of predicted RUL values.
    upper_bound_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of upper bounds for predicted RUL.
    lower_bound_dict : dict[str, List[float]]
        A dictionary where each key is a trajectory name and the value is the list of lower bounds for predicted RUL.
    true_rul_dict : dict[str, int]
        A dictionary where each key is a trajectory name and the value is the true RUL for that trajectory.

    Returns
    -------
    combined_df : pd.DataFrame
        A DataFrame combining RMSE, coverage, and WSU for each trajectory, including the average values.
    """
    df_rmse = get_rmse(mean_rul_dict, true_rul_dict)
    df_coverage = get_coverage(upper_bound_dict, lower_bound_dict, true_rul_dict)
    df_wsu = get_wsu(upper_bound_dict, lower_bound_dict)
    # Merge the dataframes on the 'name' column
    combined_df = pd.merge(df_rmse, df_coverage, on='Name')
    combined_df = pd.merge(combined_df, df_wsu, on='Name')
    return combined_df


@jit(nopython=True)
def baumwelch_method(n_states, n_obs_symbols, logPseq, fs, bs, scale, score, history, tr, emi, calc_tr, calc_emi):
    """
    Implements the Baum-Welch algorithm for parameter estimation in Hidden Markov Models (HMM).

    Parameters
    ----------
    n_states : int
        The number of hidden states in the model.
    n_obs_symbols : int
        The number of observation symbols
    logPseq : float
        The log-probability of the observed sequence.
    fs : np.ndarray
        The forward probabilities matrix (shape: [n_states, sequence_length]).
    bs : np.ndarray
        The backward probabilities matrix (shape: [n_states, sequence_length]).
    scale : np.ndarray
        The scale factors for normalization (shape: [1, sequence_length]).
    score : float
        The cumulative score (log probability) to be updated.
    history : List[int]
        The sequence of observed symbols (integer indices).
    tr : np.ndarray
        The transition matrix (shape: [n_states, n_states]).
    emi : np.ndarray
        The emission matrix (shape: [n_states, n_obs_symbols]).
    calc_tr : np.ndarray
        A precomputed matrix of transition probabilities (shape: [n_states, n_states]).
    calc_emi : np.ndarray
        A precomputed matrix of emission probabilities (shape: [n_states, n_obs_symbols]).

    Returns
    -------
    tr : np.ndarray
        Updated transition matrix after the algorithm has performed parameter estimation.
    emi : np.ndarray
        Updated emission matrix after the algorithm has performed parameter estimation.
    """
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
    """
    Computes the forward probabilities (fs) for a given sequence using the emission and transition matrices.

    Parameters
    ----------
    n_states : int
        The number of hidden states in the model.
    end_traj : int
        The length of the observation sequence.
    fs : np.ndarray
        The forward probabilities matrix (shape: [n_states, end_traj]).
    s : np.ndarray
        Scaling factors to prevent underflow (shape: [1, end_traj]).
    history : List[int]
        The sequence of observed symbols (integer indices).
    calc_emi : np.ndarray
        A matrix of emission probabilities (shape: [n_states, n_obs_symbols]).
    calc_tr : np.ndarray
        A matrix of transition probabilities (shape: [n_states, n_states]).

    Returns
    -------
    fs : np.ndarray
        The updated forward probabilities matrix.
    s : np.ndarray
        The updated scaling factors.
    """
    for count in range(1, end_traj):
        for state in range(n_states):
            fs[state, count] = calc_emi[state, history[count] - 1] * np.sum(fs[:, count - 1] * calc_tr[:, state])
        # scale factor normalizes sum(fs,count) to be 1.
        s[0, count] = np.sum(fs[:, count])
        fs[:, count] = fs[:, count] / s[0, count]
    return fs, s


@jit(nopython=True)
def bs_calculation(n_states, end_traj, bs, s, history, calc_emi, calc_tr):
    """
    Computes the backward probabilities (bs) for a given sequence using the emission and transition matrices.

    Parameters
    ----------
    n_states : int
        The number of hidden states in the model.
    end_traj : int
        The length of the observation sequence.
    bs : np.ndarray
        The backward probabilities matrix (shape: [n_states, end_traj]).
    s : np.ndarray
        Scaling factors for normalization (shape: [1, end_traj]).
    history : List[int]
        The sequence of observed symbols (integer indices).
    calc_emi : np.ndarray
        A matrix of emission probabilities (shape: [n_states, n_obs_symbols]).
    calc_tr : np.ndarray
        A matrix of transition probabilities (shape: [n_states, n_states]).

    Returns
    -------
    bs : np.ndarray
        The updated backward probabilities matrix.
    """
    for count in range(end_traj - 2, -1, -1):
        for state in range(n_states):
            bs[state, count] = (1 / s[0, count + 1]) * np.sum(
                calc_tr[state, :].T * bs[:, count + 1] * calc_emi[:, history[count + 1] - 1])
    return bs


def calculate_expected_value(pmf_values):
    """
    Calculates the expected value of a probability mass function (PMF).

    Parameters
    ----------
    pmf_values : List[float]
        A list of probabilities for each possible value.

    Returns
    -------
    expected_value : float
        The expected value calculated from the PMF.
    """
    expected_value = sum(x * p for x, p in enumerate(pmf_values))
    return expected_value


def calculate_cdf(pmf, confidence_level):
    """
    Calculates the cumulative distribution function (CDF) and percentile values for a given probability mass function (PMF).

    Parameters
    ----------
    pmf : List[float]
        A list of probabilities for each possible value.
    confidence_level : float
        The confidence level for calculating the percentiles (e.g., 0.95 for 95%).

    Returns
    -------
    lower_value : int
        The index corresponding to the lower percentile.
    """
    cdf = np.cumsum(pmf)
    # Calculate the lower and upper percentiles
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    lower_value = np.argmax(cdf >= lower_percentile)
    upper_value = np.argmax(cdf >= upper_percentile)

    return lower_value, upper_value


def create_folders():
    """
    Create a directory structure for storing results.

    This function creates a main "results" folder in the current working directory 
    and subdirectories within it, including "dictionaries", "figures", and "models". 
    If the folders already exist, a message is printed indicating so.

    Notes
    -----
    - The function does not take any parameters.
    - The function does not return any values.
    - The created folder structure is as follows:
      
      results/

      ├── dictionaries/

      ├── figures/

      ├── models/

    Examples
    --------
    >>> create_folders()
    Created folder: /path/to/current/directory/results
    Created folder: /path/to/current/directory/results/dictionaries
    Created folder: /path/to/current/directory/results/figures
    Created folder: /path/to/current/directory/results/models
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
