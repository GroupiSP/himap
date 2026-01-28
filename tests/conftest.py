# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def himap_modules():
    """
    Returns (himap, base, utils) once import succeeds.
    """
    base = importlib.import_module("himap.base")
    utils = importlib.import_module("himap.utils")
    himap = importlib.import_module("himap")
    return himap, base, utils


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """
    Run each test in an isolated working directory so ./results is created in tmp, not the repo.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def no_tqdm(monkeypatch, himap_modules):
    """
    Disable tqdm progress bars inside himap.base for cleaner test output.
    """
    _, base, _ = himap_modules

    def _tqdm(it, **kwargs):
        return it

    monkeypatch.setattr(base, "tqdm", _tqdm)


def _subset_reindex(seqs: dict, n_traj: int):
    """
    Take first n_traj trajectories.
    Reindex to traj_0..traj_{n-1} (required by HMM.fit which indexes by traj_{i}).
    """
    keys = list(seqs.keys())[:n_traj]
    out = {}
    for i, k in enumerate(keys):
        s = list(seqs[k])
        out[f"traj_{i}"] = s
    return out


@pytest.fixture()
def cmapss_train_test(himap_modules):
    """
    Load CMAPSS example data directly from himap/example_data (no cwd dependency).
    Returns (train_dict, test_dict, f_value, obs_state_len).
    """
    _, _, utils = himap_modules
    pkg_dir = Path(utils.__file__).resolve().parent
    data_dir = pkg_dir / "example_data"

    f_value = 21
    obs_state_len = 5

    train_path = data_dir / "train_FD001_disc_20_mod.csv"
    test_path = data_dir / "test_FD001_disc_20_mod.csv"

    train_df = pd.read_csv(train_path, sep=";")
    test_df = pd.read_csv(test_path, sep=";")

    def build_dict(df):
        units = np.unique(df["unit_nr"].to_numpy())
        seqs = {}
        for i, unit in enumerate(units):
            seq = df.loc[df["unit_nr"] == unit]["s_discretized"].to_numpy() + 1
            failure = np.array([f_value] * obs_state_len)
            seq = np.concatenate([seq, failure]).tolist()
            seqs[f"traj_{i}"] = seq
        return seqs

    return build_dict(train_df), build_dict(test_df), f_value, obs_state_len


@pytest.fixture()
def cmapss_small(cmapss_train_test):
    """
    A small subset for fast tests.
    """
    train, test, f_value, obs_state_len = cmapss_train_test
    train_small = _subset_reindex(train, n_traj=3)
    test_small = _subset_reindex(test, n_traj=1)
    return train_small, test_small, f_value, obs_state_len
