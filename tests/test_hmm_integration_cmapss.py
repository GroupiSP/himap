# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors

import numpy as np
import pandas as pd


def test_hmm_cmapss_fit_and_metrics_are_sane(workdir, no_tqdm, himap_modules, cmapss_small):
    _, base, _ = himap_modules
    train_small, test_small, f_value, _ = cmapss_small

    # Force quick "convergence" so tr/emi are set (HMM.fit sets them only on converge)
    hmm = base.HMM(
        n_states=6,
        n_obs_symbols=f_value,   # f_value=21 (observations are 1..21)
        name="pytest_hmm_cmapss",
    )

    hmm, scores = hmm.fit(train_small, return_all_scores=True, save_iters=False)

    assert len(scores) >= 1, "Expected at least one iteration score"
    assert np.all(np.isfinite(scores)), f"Non-finite HMM scores: {scores}"

    # Run prognostics on a tiny test subset, no plots (fast + headless-friendly)
    hmm.prognostics(test_small, max_samples=5000, plot_rul=False, get_metrics=True)

    # Validate metrics output
    df_path = workdir / "results" / "df_results.csv"
    assert df_path.exists(), f"Expected metrics CSV at {df_path}"

    df = pd.read_csv(df_path)
    needed = {"Name", "rmse", "coverage", "wsu"}
    assert needed.issubset(df.columns), f"Missing columns. Got: {list(df.columns)}"

    # "Make sense" checks - broad and stable
    assert np.all(np.isfinite(df["rmse"])), f"Non-finite rmse:\n{df}"
    assert np.all(df["rmse"] >= 0), f"Negative rmse:\n{df}"

    assert np.all(np.isfinite(df["coverage"])), f"Non-finite coverage:\n{df}"
    # coverage should be in [0,1] if bounds/true values behave normally
    assert np.all((df["coverage"] >= 0) & (df["coverage"] <= 1)), f"Coverage outside [0,1]:\n{df}"

    assert np.all(np.isfinite(df["wsu"])), f"Non-finite wsu:\n{df}"
    assert np.all(df["wsu"] >= 0), f"Negative wsu:\n{df}"
