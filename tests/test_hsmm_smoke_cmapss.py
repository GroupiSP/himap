# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors

import numpy as np
import pytest


# Skip this whole file if the extension module cannot be imported
pytest.importorskip(
    "himap.cython_build.fwd_bwd",
    reason="Skipping HSMM tests - HiMAP Cython extension not built. "
           "Build with: python setup.py build_ext --inplace (or pip install -e .)"
)

def test_hsmm_two_iters_loglik_is_finite(workdir, no_tqdm, himap_modules, cmapss_small):
    _, base, _ = himap_modules
    train_small, _, f_value, obs_state_len = cmapss_small

    hsmm = base.GaussianHSMM(
        n_states=6,
        n_durations=200,      # > obs_state_len
        n_iter=2,            # "two epochs"
        obs_state_len=obs_state_len,
        f_value=f_value,
        random_state=0,
        name="pytest_hsmm_cmapss",
    )

    hsmm.fit(train_small)

    # fit can break early, so we just require at least one iter and finiteness
    assert hasattr(hsmm, "score_per_iter"), "HSMM did not store score_per_iter"
    assert hsmm.score_per_iter.size >= 1, "Expected at least one HSMM score"
    assert np.all(np.isfinite(hsmm.score_per_iter)), f"Non-finite HSMM score_per_iter: {hsmm.score_per_iter.ravel()}"

    # Optional extra sanity: parameters are finite after training
    assert np.all(np.isfinite(hsmm.mean)), "HSMM mean has NaN/inf"
    assert np.all(np.isfinite(hsmm.covmat)), "HSMM cov has NaN/inf"
