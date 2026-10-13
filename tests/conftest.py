"""
Shared pytest fixtures for the RF data preprocessing pipeline test suite.

All fixtures use synthetic data only — no real dataset files required.
"""

import numpy as np
import pytest


@pytest.fixture
def complex64_bin_file(tmp_path):
    """
    Write 4 known complex64 samples to a .bin file.

    The values are chosen to be easily verifiable:
      sample 0: I=1.0, Q=2.0
      sample 1: I=3.0, Q=4.0
      sample 2: I=5.0, Q=6.0
      sample 3: I=7.0, Q=8.0
    """
    iq = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
    path = tmp_path / "test_complex64.bin"
    iq.tofile(path)
    return path, iq


@pytest.fixture
def int16_bin_file(tmp_path):
    """
    Write interleaved int16 I,Q pairs to a .bin file.

    Layout: [I0, Q0, I1, Q1, ...] — 4 samples total.
    Values are well below int16 max (32767) so normalization stays in [-1, 1].
    """
    raw = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int16)
    path = tmp_path / "test_int16.bin"
    raw.tofile(path)
    return path, raw
