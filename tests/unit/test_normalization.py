"""
Tests for pipeline/preprocessing.py :: normalize_window

Covers all five normalization modes. "unit_power" gets the heaviest coverage:
it is the frozen v1.0 default applied to every window in the released corpus,
so a silent regression here corrupts the whole collection while leaving every
shape and dtype check green.

Synthetic data only — no dataset files required.
"""

import numpy as np
import pytest

from pipeline.preprocessing import compute_window_power_lin, normalize_window


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_window(i_value: float, q_value: float, samples: int = 64,
                dtype=np.float32) -> np.ndarray:
    """Build a constant (2, samples) window with distinct I and Q levels."""
    window = np.empty((2, samples), dtype=dtype)
    window[0] = i_value
    window[1] = q_value
    return window


def make_random_window(samples: int = 256, scale: float = 1.0,
                       seed: int = 0, dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, samples)) * scale).astype(dtype)


# ── unit_power: the frozen v1.0 default ───────────────────────────────────────

class TestUnitPower:

    def test_output_power_is_unity(self):
        """mean(I² + Q²) must be 1.0 after normalization."""
        window = make_random_window(256, scale=17.0)
        out = normalize_window(window, "unit_power")
        assert compute_window_power_lin(out) == pytest.approx(1.0, rel=1e-5)

    def test_power_is_joint_not_per_channel(self):
        """
        Normalization divides the whole window by one scalar — it does NOT
        normalize I and Q independently.

        I=3, Q=4 → power = mean(9+16) = 25 → scale = 5 → I=0.6, Q=0.8.
        A per-channel implementation would instead give I=1.0, Q=1.0
        (and a joint power of 2.0), so these constants tell the two apart.
        """
        window = make_window(3.0, 4.0)
        out = normalize_window(window, "unit_power")

        np.testing.assert_allclose(out[0], 0.6, rtol=1e-6)
        np.testing.assert_allclose(out[1], 0.8, rtol=1e-6)
        assert compute_window_power_lin(out) == pytest.approx(1.0, rel=1e-6)

    def test_iq_ratio_preserved(self):
        """A single scalar divisor must leave the I/Q ratio untouched."""
        window = make_random_window(256, scale=3.0, seed=7)
        out = normalize_window(window, "unit_power")

        ratio_in = window[0] / window[1]
        ratio_out = out[0] / out[1]
        np.testing.assert_allclose(ratio_in, ratio_out, rtol=1e-4)

    def test_scale_invariance(self):
        """Amplitude-scaled copies of a signal normalize to the same window."""
        base = make_random_window(256, scale=1.0, seed=3)
        loud = (base * 1000.0).astype(np.float32)

        np.testing.assert_allclose(
            normalize_window(base, "unit_power"),
            normalize_window(loud, "unit_power"),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_idempotent(self):
        """Normalizing an already-normalized window is a no-op."""
        once = normalize_window(make_random_window(256, scale=5.0), "unit_power")
        twice = normalize_window(once, "unit_power")
        np.testing.assert_allclose(once, twice, rtol=1e-5)

    def test_zero_power_window_returned_unchanged(self):
        """An all-zero window has no power to normalize — it must pass through."""
        window = np.zeros((2, 64), dtype=np.float32)
        out = normalize_window(window, "unit_power")
        np.testing.assert_array_equal(out, np.zeros((2, 64), dtype=np.float32))

    def test_zero_power_window_produces_no_nan_or_inf(self):
        """The zero-power guard exists to stop 0/0 — verify it actually does."""
        out = normalize_window(np.zeros((2, 64), dtype=np.float32), "unit_power")
        assert np.all(np.isfinite(out))

    def test_tiny_but_nonzero_power_stays_finite(self):
        """Denormal-scale input must not overflow to inf through the 1e-12 epsilon."""
        window = np.full((2, 64), 1e-20, dtype=np.float32)
        out = normalize_window(window, "unit_power")
        assert np.all(np.isfinite(out))

    def test_dtype_preserved_float32(self):
        window = make_random_window(64, dtype=np.float32)
        assert normalize_window(window, "unit_power").dtype == np.float32

    def test_dtype_preserved_float64(self):
        window = make_random_window(64, dtype=np.float64)
        assert normalize_window(window, "unit_power").dtype == np.float64

    def test_shape_preserved(self):
        window = make_random_window(500)
        assert normalize_window(window, "unit_power").shape == (2, 500)

    def test_input_not_mutated(self):
        """Normalization must not modify the caller's array in place."""
        window = make_random_window(128, scale=9.0)
        before = window.copy()
        normalize_window(window, "unit_power")
        np.testing.assert_array_equal(window, before)


# ── Pass-through modes ────────────────────────────────────────────────────────

class TestNoNormalization:

    @pytest.mark.parametrize("mode", [None, "none"])
    def test_values_unchanged(self, mode):
        window = make_random_window(64, scale=42.0)
        np.testing.assert_array_equal(normalize_window(window, mode), window)

    @pytest.mark.parametrize("mode", [None, "none"])
    def test_dtype_preserved(self, mode):
        window = make_random_window(64, dtype=np.float64)
        assert normalize_window(window, mode).dtype == np.float64


# ── max_magnitude ─────────────────────────────────────────────────────────────

class TestMaxMagnitude:

    def test_peak_becomes_unity(self):
        window = make_random_window(256, scale=13.0)
        out = normalize_window(window, "max_magnitude")
        assert np.max(np.abs(out)) == pytest.approx(1.0, rel=1e-5)

    def test_scales_across_both_channels(self):
        """The divisor is the global max over I and Q, not a per-channel max."""
        window = make_window(2.0, 8.0)
        out = normalize_window(window, "max_magnitude")
        np.testing.assert_allclose(out[0], 0.25, rtol=1e-6)
        np.testing.assert_allclose(out[1], 1.0, rtol=1e-6)

    def test_zero_window_returned_unchanged(self):
        window = np.zeros((2, 32), dtype=np.float32)
        np.testing.assert_array_equal(
            normalize_window(window, "max_magnitude"), window
        )

    def test_dtype_preserved(self):
        window = make_random_window(64, dtype=np.float32)
        assert normalize_window(window, "max_magnitude").dtype == np.float32


# ── zero_mean_unit_variance ───────────────────────────────────────────────────

class TestZeroMeanUnitVariance:

    def test_per_channel_zero_mean(self):
        """Standardization is per-channel: each of I and Q gets mean 0."""
        window = make_random_window(512, scale=4.0, seed=11)
        window[1] += 100.0  # give Q a large offset the I channel does not have
        out = normalize_window(window, "zero_mean_unit_variance")

        assert np.mean(out[0]) == pytest.approx(0.0, abs=1e-5)
        assert np.mean(out[1]) == pytest.approx(0.0, abs=1e-5)

    def test_per_channel_unit_variance(self):
        window = make_random_window(512, scale=4.0, seed=11)
        window[1] *= 50.0  # different spread per channel
        out = normalize_window(window, "zero_mean_unit_variance")

        assert np.std(out[0]) == pytest.approx(1.0, rel=1e-4)
        assert np.std(out[1]) == pytest.approx(1.0, rel=1e-4)

    def test_constant_channel_does_not_divide_by_zero(self):
        """A flat channel has zero std; the guard must keep the output finite."""
        window = make_window(5.0, 5.0)
        out = normalize_window(window, "zero_mean_unit_variance")
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_dtype_preserved(self):
        window = make_random_window(64, dtype=np.float32)
        assert normalize_window(window, "zero_mean_unit_variance").dtype == np.float32


# ── mapminmax ─────────────────────────────────────────────────────────────────

class TestMapMinMax:

    def test_per_channel_range_is_minus_one_to_one(self):
        window = make_random_window(512, scale=7.0, seed=5)
        out = normalize_window(window, "mapminmax")

        for channel in range(2):
            assert np.min(out[channel]) == pytest.approx(-1.0, abs=1e-5)
            assert np.max(out[channel]) == pytest.approx(1.0, abs=1e-5)

    def test_channels_scaled_independently(self):
        """Each channel is mapped from its own [min, max], not a shared one."""
        window = np.empty((2, 4), dtype=np.float32)
        window[0] = [0.0, 1.0, 2.0, 4.0]      # spans 0..4
        window[1] = [-10.0, -5.0, 0.0, 10.0]  # spans -10..10
        out = normalize_window(window, "mapminmax")

        np.testing.assert_allclose(out[0], [-1.0, -0.5, 0.0, 1.0], rtol=1e-6)
        np.testing.assert_allclose(out[1], [-1.0, -0.5, 0.0, 1.0], rtol=1e-6)

    def test_constant_channel_maps_to_midpoint(self):
        """Zero span cannot be scaled; the channel collapses to the midpoint 0.0."""
        window = make_window(3.0, 3.0)
        out = normalize_window(window, "mapminmax")
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_dtype_preserved(self):
        window = make_random_window(64, dtype=np.float32)
        assert normalize_window(window, "mapminmax").dtype == np.float32


# ── Rejection of unknown modes ────────────────────────────────────────────────

class TestUnknownMode:

    def test_unknown_mode_raises(self):
        """
        An unrecognized mode must raise, not silently pass the window through.
        A typo in a dataset config should fail the build, not quietly ship
        unnormalized data.
        """
        with pytest.raises(ValueError, match="Unknown normalize mode"):
            normalize_window(make_random_window(32), "unitpower")
