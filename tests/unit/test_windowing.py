"""
Tests for pipeline/windowing.py

Covers:
- iter_windows_from_bin: dtype correctness (the critical regression guard)
- window_iq_signal: shape, padding, metadata
"""

import numpy as np
import pytest

from pipeline.windowing import iter_windows_from_bin, window_iq_signal


# ── iter_windows_from_bin: complex64 ──────────────────────────────────────────

class TestIterWindowsFromBinComplex64:
    """
    Tests for complex64 reading.

    Reading a complex64 file with an int16 dtype raises no error and yields
    imbalanced I/Q data. These tests pin the correct reading.
    """

    def test_iq_values_correct(self, complex64_bin_file):
        """I and Q values round-trip exactly through the bin reader."""
        path, source = complex64_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.complex64
        ))
        assert len(windows) == 1
        window, _ = windows[0]  # shape (2, 4)
        expected_i = source.real.astype(np.float32)
        expected_q = source.imag.astype(np.float32)
        np.testing.assert_array_almost_equal(window[0], expected_i)
        np.testing.assert_array_almost_equal(window[1], expected_q)

    def test_output_dtype_is_float32(self, complex64_bin_file):
        path, _ = complex64_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.complex64
        ))
        window, _ = windows[0]
        assert window.dtype == np.float32

    def test_output_shape(self, complex64_bin_file):
        """4 complex samples → one window of shape (2, 4)."""
        path, _ = complex64_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.complex64
        ))
        assert len(windows) == 1
        window, _ = windows[0]
        assert window.shape == (2, 4)

    def test_iq_ratio_is_balanced(self, complex64_bin_file):
        """
        I and Q amplitudes are similar.
        Reading a complex64 file as int16 produces I≈0, Q-dominant output.
        """
        path, _ = complex64_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.complex64
        ))
        window, _ = windows[0]
        i_power = np.mean(window[0] ** 2)
        q_power = np.mean(window[1] ** 2)
        ratio = i_power / (q_power + 1e-9)
        # With correct dtype, I and Q powers should be in the same ballpark (ratio ~1)
        assert 0.1 < ratio < 10.0, (
            f"I/Q power ratio {ratio:.4f} suggests a dtype mismatch. "
            f"Expected ratio near 1.0 for valid complex64 data."
        )

# ── iter_windows_from_bin: int16 ──────────────────────────────────────────────

class TestIterWindowsFromBinInt16:

    def test_normalization_range(self, int16_bin_file):
        """int16 values must be normalized to [-1, 1]."""
        path, _ = int16_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.int16
        ))
        assert len(windows) == 1
        window, _ = windows[0]
        assert np.all(window >= -1.0)
        assert np.all(window <= 1.0)

    def test_output_dtype_is_float32(self, int16_bin_file):
        path, _ = int16_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.int16
        ))
        window, _ = windows[0]
        assert window.dtype == np.float32

    def test_output_shape(self, int16_bin_file):
        """8 int16 values (4 I,Q pairs) → one window of shape (2, 4)."""
        path, _ = int16_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.int16
        ))
        assert len(windows) == 1
        window, _ = windows[0]
        assert window.shape == (2, 4)

    def test_normalization_values(self, int16_bin_file):
        """Check specific normalized values against expected float equivalents."""
        path, raw = int16_bin_file
        windows = list(iter_windows_from_bin(
            path, window_size=4, hop_size=4, pad_last=False, dtype=np.int16
        ))
        window, _ = windows[0]
        int16_max = np.iinfo(np.int16).max  # 32767
        expected_i = raw[::2].astype(np.float32) / int16_max  # [100, 300, 500, 700] / 32767
        np.testing.assert_array_almost_equal(window[0], expected_i, decimal=5)


# ── window_iq_signal ──────────────────────────────────────────────────────────

class TestWindowIqSignal:

    def test_output_shape_non_overlapping(self):
        """1000 samples, window=500, hop=500 → 2 windows of shape (2, 500)."""
        iq = np.zeros((1000, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=False)
        assert windows.shape == (2, 2, 500)

    def test_output_shape_overlapping(self):
        """1000 samples, window=500, hop=250 → 3 windows."""
        iq = np.zeros((1000, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=250, pad_last=False)
        assert windows.shape[0] == 3

    def test_pad_last_true_adds_padded_window(self):
        """600 samples, window=500, hop=500 → 2 windows (second is zero-padded)."""
        iq = np.ones((600, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=True)
        assert windows.shape[0] == 2

    def test_pad_last_false_drops_tail(self):
        """600 samples, window=500, hop=500 → 1 window (tail dropped)."""
        iq = np.ones((600, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=False)
        assert windows.shape[0] == 1

    def test_padded_flag_set_correctly(self):
        """Only the last (incomplete) window should have padded=True."""
        iq = np.ones((600, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=True)
        assert meta["padded"][0] == False
        assert meta["padded"][1] == True

    def test_start_idx_correct(self):
        """start_idx should be [0, 500] for non-overlapping windows."""
        iq = np.zeros((1000, 2), dtype=np.float32)
        _, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=False)
        np.testing.assert_array_equal(meta["start_idx"], [0, 500])

    def test_padded_region_is_zeros(self):
        """The zero-padded tail of the last window should be all zeros."""
        iq = np.ones((600, 2), dtype=np.float32)
        windows, _ = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=True)
        # Second window: first 100 samples are signal (=1.0), rest are padding (=0.0)
        last_window = windows[1]  # shape (2, 500)
        np.testing.assert_array_equal(last_window[:, 100:], 0.0)

    def test_empty_signal_pad_last_false(self):
        """Empty signal returns zero windows."""
        iq = np.zeros((0, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=False)
        assert windows.shape[0] == 0

    def test_signal_shorter_than_window_pad_true(self):
        """Signal shorter than window_size → single zero-padded window if pad_last=True."""
        iq = np.ones((50, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=True)
        assert windows.shape == (1, 2, 500)
        assert meta["padded"][0] == True
        np.testing.assert_array_equal(windows[0, :, 50:], 0.0)

    def test_signal_shorter_than_window_pad_false(self):
        """Signal shorter than window_size → no windows if pad_last=False."""
        iq = np.ones((50, 2), dtype=np.float32)
        windows, meta = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=False)
        assert windows.shape[0] == 0

    def test_invalid_shape_raises(self):
        """Non-[N,2] input raises ValueError."""
        iq = np.zeros((100, 3))
        with pytest.raises(ValueError, match="Expected IQ shape"):
            window_iq_signal(iq, window_size=50, hop_size=50, pad_last=False)

    def test_output_dtype_preserved(self):
        """Output dtype matches input dtype."""
        iq = np.zeros((1000, 2), dtype=np.float32)
        windows, _ = window_iq_signal(iq, window_size=500, hop_size=500, pad_last=False)
        assert windows.dtype == np.float32
