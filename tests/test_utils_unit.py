# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Group iSP and contributors

import numpy as np
import pytest


def test_create_folders_creates_expected_tree(workdir, himap_modules):
    _, _, utils = himap_modules

    utils.create_folders()

    assert (workdir / "results").is_dir()
    for sub in ["dictionaries", "figures", "models"]:
        assert (workdir / "results" / sub).is_dir(), f"Missing folder: results/{sub}"


@pytest.mark.parametrize(
    "text,expected",
    [("yes", True), ("true", True), ("t", True), ("1", True),
     ("no", False), ("false", False), ("f", False), ("0", False)],
)
def test_str2bool(text, expected, himap_modules):
    _, _, utils = himap_modules
    assert utils.str2bool(text) is expected


def test_log_mask_zero(himap_modules):
    _, _, utils = himap_modules
    a = np.array([1.0, 0.0, 10.0])
    out = utils.log_mask_zero(a)

    assert np.isfinite(out[0])
    assert np.isneginf(out[1])
    assert np.isclose(out[2], np.log(10.0))
