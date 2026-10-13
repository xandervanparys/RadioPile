"""
Tests for pipeline/preprocessing.py noise-floor detection and filtering.

These code paths DELETE DATA: a wrong threshold either keeps pure noise or
discards real signal, and neither shows up in a shape or dtype check. Every
threshold assertion here is against a constructed distribution with an
arithmetically known answer rather than a recorded output.

Synthetic data only — no dataset files required.
"""

import numpy as np
import pytest

from pipeline.preprocessing import (
    compute_window_power_db,
    compute_window_power_lin,
    detect_noise_floor_from_windows,
    detect_noise_floor_histogram,
    detect_noise_floor_percentile,
    detect_noise_floor_statistical,
    determine_noise_threshold,
    filter_window_by_power,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def window_with_power_db(target_db: float, samples: int = 64) -> np.ndarray:
    """
    Build a constant window whose measured power is exactly target_db.

    power_lin = mean(I² + Q²); with I = Q = a this is 2a², so a = sqrt(p/2).
    """
    power_lin = 10.0 ** (target_db / 10.0)
    amplitude = np.sqrt(power_lin / 2.0)
    return np.full((2, samples), amplitude, dtype=np.float64)


def windows_iter(powers_db):
    """Yield (window, meta) pairs with the given powers, as the pipeline does."""
    for i, power_db in enumerate(powers_db):
        yield window_with_power_db(power_db), {"index": i}


# ── Power measurement (the basis of every threshold decision) ─────────────────

class TestWindowPower:

    def test_power_lin_is_mean_of_squares(self):
        """I=3, Q=4 → mean(9 + 16) = 25."""
        window = np.empty((2, 16), dtype=np.float64)
        window[0] = 3.0
        window[1] = 4.0
        assert compute_window_power_lin(window) == pytest.approx(25.0)

    def test_power_db_matches_linear(self):
        """10·log10(25) ≈ 13.979 dB."""
        window = np.empty((2, 16), dtype=np.float64)
        window[0] = 3.0
        window[1] = 4.0
        assert compute_window_power_db(window) == pytest.approx(13.9794, abs=1e-3)

    def test_helper_round_trips_through_power_db(self):
        """The test helper must actually produce the power it claims."""
        for target in (-60.0, -40.0, 0.0, 12.5):
            assert compute_window_power_db(window_with_power_db(target)) == \
                pytest.approx(target, abs=1e-4)

    def test_zero_window_does_not_return_negative_infinity(self):
        """The +1e-12 guard keeps an all-zero window finite rather than -inf."""
        floor = compute_window_power_db(np.zeros((2, 16)))
        assert np.isfinite(floor)
        assert floor < -100.0

    def test_epsilon_biases_very_low_powers_upward(self):
        """
        compute_window_power_db adds 1e-12 to the linear power before taking the
        log. That guard is invisible at realistic signal levels but measurably
        lifts very quiet windows, so thresholds set near the bottom of the range
        are slightly conservative (they keep marginally more than they appear to).

        The synthetic noise floors used elsewhere in this file sit at -60 dB,
        where the bias is ~4e-6 dB and the arithmetic below is exact.
        """
        assert compute_window_power_db(window_with_power_db(-60.0)) == \
            pytest.approx(-60.0, abs=1e-4)
        # Two decades quieter, the same guard is worth several dB.
        assert compute_window_power_db(window_with_power_db(-120.0)) > -119.0


# ── Percentile detector ───────────────────────────────────────────────────────

class TestPercentileDetector:

    def test_exact_percentile_on_linear_ramp(self):
        """
        0..100 inclusive has 101 points, so the p-th percentile lands exactly on
        index (n-1)·p/100 = p — no interpolation ambiguity.
        """
        powers = np.arange(0.0, 101.0)
        assert detect_noise_floor_percentile(powers, percentile=5.0) == pytest.approx(5.0)
        assert detect_noise_floor_percentile(powers, percentile=25.0) == pytest.approx(25.0)

    def test_percentile_zero_is_minimum(self):
        powers = np.array([-95.0, -80.0, -20.0, 5.0])
        assert detect_noise_floor_percentile(powers, percentile=0.0) == pytest.approx(-95.0)

    def test_percentile_hundred_is_maximum(self):
        powers = np.array([-95.0, -80.0, -20.0, 5.0])
        assert detect_noise_floor_percentile(powers, percentile=100.0) == pytest.approx(5.0)

    def test_ignores_signal_heavy_tail(self):
        """A low percentile must track the noise cluster, not the loud windows."""
        powers = np.concatenate([np.full(10, -95.0), np.full(90, -10.0)])
        assert detect_noise_floor_percentile(powers, percentile=5.0) == pytest.approx(-95.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty array"):
            detect_noise_floor_percentile(np.array([]))


# ── Histogram detector ────────────────────────────────────────────────────────

class TestHistogramDetector:

    def test_returns_mode_of_noise_region(self):
        """
        The lowest 10% are all exactly -95 dB, so the histogram mode of the
        noise region is -95 (within one bin width of the degenerate range
        numpy uses for a constant array).
        """
        powers = np.concatenate([np.full(10, -95.0), np.full(90, -10.0)])
        floor = detect_noise_floor_histogram(powers, bins=100, noise_fraction=0.1)
        assert floor == pytest.approx(-95.0, abs=0.02)

    def test_noise_fraction_selects_the_analyzed_region(self):
        """
        Widening noise_fraction pulls higher-power windows into the region and
        must move the estimate upward.
        """
        powers = np.concatenate([np.full(10, -95.0), np.full(90, -10.0)])
        narrow = detect_noise_floor_histogram(powers, bins=50, noise_fraction=0.1)
        wide = detect_noise_floor_histogram(powers, bins=50, noise_fraction=1.0)
        assert wide > narrow

    def test_picks_denser_mode_when_two_clusters_share_the_region(self):
        """With both clusters inside the region, the denser one wins."""
        powers = np.concatenate([np.full(5, -90.0), np.full(95, -50.0)])
        floor = detect_noise_floor_histogram(powers, bins=40, noise_fraction=1.0)
        assert floor == pytest.approx(-50.0, abs=1.0)

    def test_noise_region_never_empty(self):
        """max(1, ...) keeps a tiny input from producing an empty region."""
        powers = np.array([-70.0, -20.0])
        floor = detect_noise_floor_histogram(powers, bins=10, noise_fraction=0.01)
        assert floor == pytest.approx(-70.0, abs=0.1)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty array"):
            detect_noise_floor_histogram(np.array([]))


# ── Statistical detector ──────────────────────────────────────────────────────

class TestStatisticalDetector:

    def test_exact_mean_minus_n_std(self):
        """
        Lowest 10% of 20 points = 2 points, [0, 10]: mean 5, std 5,
        so mean − 2·std = −5 exactly.
        """
        powers = np.concatenate([np.array([0.0, 10.0]), np.full(18, 100.0)])
        floor = detect_noise_floor_statistical(
            powers, noise_fraction=0.1, std_multiplier=2.0
        )
        assert floor == pytest.approx(-5.0)

    def test_std_multiplier_scales_the_offset(self):
        powers = np.concatenate([np.array([0.0, 10.0]), np.full(18, 100.0)])
        assert detect_noise_floor_statistical(
            powers, noise_fraction=0.1, std_multiplier=0.0
        ) == pytest.approx(5.0)
        assert detect_noise_floor_statistical(
            powers, noise_fraction=0.1, std_multiplier=1.0
        ) == pytest.approx(0.0)

    def test_constant_noise_region_returns_that_value(self):
        """Zero spread means the multiplier has nothing to subtract."""
        powers = np.concatenate([np.full(10, -88.0), np.full(90, -10.0)])
        floor = detect_noise_floor_statistical(
            powers, noise_fraction=0.1, std_multiplier=3.0
        )
        assert floor == pytest.approx(-88.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty array"):
            detect_noise_floor_statistical(np.array([]))


# ── Detector dispatch over a window iterator ──────────────────────────────────

class TestDetectFromWindows:

    def test_percentile_dispatch(self):
        powers = [-60.0] * 10 + [-10.0] * 90
        floor = detect_noise_floor_from_windows(
            windows_iter(powers), method="percentile", percentile=5.0
        )
        assert floor == pytest.approx(-60.0, abs=1e-4)

    def test_statistical_dispatch(self):
        powers = [-55.0] * 10 + [-10.0] * 90
        floor = detect_noise_floor_from_windows(
            windows_iter(powers), method="statistical",
            noise_fraction=0.1, std_multiplier=3.0,
        )
        assert floor == pytest.approx(-55.0, abs=1e-4)

    def test_histogram_dispatch(self):
        powers = [-60.0] * 10 + [-10.0] * 90
        floor = detect_noise_floor_from_windows(
            windows_iter(powers), method="histogram",
            bins=100, noise_fraction=0.1,
        )
        assert floor == pytest.approx(-60.0, abs=0.05)

    def test_max_samples_truncates_the_scan(self):
        """
        Only the first max_samples windows are consulted. Feeding quiet windows
        first and loud ones after, a truncated scan must see only the quiet ones.
        """
        powers = [-60.0] * 10 + [0.0] * 990
        floor = detect_noise_floor_from_windows(
            windows_iter(powers), method="percentile",
            percentile=100.0, max_samples=10,
        )
        assert floor == pytest.approx(-60.0, abs=1e-4)

    def test_without_truncation_the_loud_windows_are_seen(self):
        """Control for the previous test: the full scan reaches the loud tail."""
        powers = [-60.0] * 10 + [0.0] * 990
        floor = detect_noise_floor_from_windows(
            windows_iter(powers), method="percentile",
            percentile=100.0, max_samples=10_000,
        )
        assert floor == pytest.approx(0.0, abs=1e-4)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown noise floor detection method"):
            detect_noise_floor_from_windows(windows_iter([-90.0]), method="magic")

    def test_no_windows_raises(self):
        with pytest.raises(ValueError, match="No windows available"):
            detect_noise_floor_from_windows(iter([]), method="percentile")


# ── determine_noise_threshold: the configuration front door ───────────────────

class TestDetermineNoiseThreshold:

    def test_none_disables_filtering(self):
        assert determine_noise_threshold(None, iter([])) is None

    def test_float_threshold_passes_through(self):
        assert determine_noise_threshold(-70.0, iter([])) == pytest.approx(-70.0)

    def test_int_threshold_coerced_to_float(self):
        result = determine_noise_threshold(-70, iter([]))
        assert isinstance(result, float)
        assert result == pytest.approx(-70.0)

    def test_manual_threshold_does_not_consume_the_iterator(self):
        """A fixed threshold must not scan windows it does not need."""
        consumed = []

        def counting_iter():
            for power in (-95.0, -10.0):
                consumed.append(power)
                yield window_with_power_db(power), {}

        determine_noise_threshold(-70.0, counting_iter())
        assert consumed == []

    def test_auto_applies_margin_above_detected_floor(self):
        """auto → detected floor (−60) + margin (6) = −54."""
        powers = [-60.0] * 10 + [-10.0] * 90
        threshold = determine_noise_threshold(
            "auto",
            windows_iter(powers),
            detection_method="percentile",
            detection_params={"percentile": 5.0, "margin_db": 6.0},
            verbose=False,
        )
        assert threshold == pytest.approx(-54.0, abs=1e-3)

    def test_margin_is_not_forwarded_to_the_detector(self):
        """
        margin_db is popped before the params are splatted into the detector.
        Were it forwarded, detect_noise_floor_percentile would raise TypeError,
        the broad except would swallow it, and this would come back None.
        """
        powers = [-60.0] * 10 + [-10.0] * 90
        threshold = determine_noise_threshold(
            "auto",
            windows_iter(powers),
            detection_method="percentile",
            detection_params={"percentile": 5.0, "margin_db": 6.0},
            verbose=False,
        )
        assert threshold is not None

    def test_auto_without_margin_returns_bare_floor(self):
        powers = [-60.0] * 10 + [-10.0] * 90
        threshold = determine_noise_threshold(
            "auto",
            windows_iter(powers),
            detection_method="percentile",
            detection_params={"percentile": 5.0},
            verbose=False,
        )
        assert threshold == pytest.approx(-60.0, abs=1e-3)

    def test_auto_with_no_detection_params(self):
        """detection_params=None must not crash; defaults apply."""
        powers = np.linspace(-60.0, 0.0, 101).tolist()
        threshold = determine_noise_threshold(
            "auto", windows_iter(powers),
            detection_method="percentile", detection_params=None, verbose=False,
        )
        assert threshold is not None

    def test_auto_statistical_method(self):
        powers = [-55.0] * 10 + [-10.0] * 90
        threshold = determine_noise_threshold(
            "auto",
            windows_iter(powers),
            detection_method="statistical",
            detection_params={"noise_fraction": 0.1, "std_multiplier": 3.0,
                              "margin_db": 2.0},
            verbose=False,
        )
        assert threshold == pytest.approx(-53.0, abs=1e-3)

    def test_detection_failure_disables_filtering_rather_than_crashing(self):
        """
        An empty iterator makes detection fail. The pipeline chooses to keep all
        windows rather than abort the build, so the threshold must come back None.
        """
        assert determine_noise_threshold(
            "auto", iter([]), detection_method="percentile", verbose=False
        ) is None

    def test_unknown_method_disables_filtering(self):
        assert determine_noise_threshold(
            "auto", windows_iter([-90.0]),
            detection_method="magic", verbose=False,
        ) is None

    def test_invalid_string_raises(self):
        """Anything other than None / float / 'auto' is a config error."""
        with pytest.raises(ValueError, match="Invalid noise_power_threshold_db"):
            determine_noise_threshold("yes please", iter([]))

# ── filter_window_by_power ────────────────────────────────────────────────────

class TestFilterWindowByPower:

    def test_window_above_threshold_kept(self):
        assert filter_window_by_power(window_with_power_db(-50.0), -70.0)

    def test_window_below_threshold_dropped(self):
        assert not filter_window_by_power(window_with_power_db(-90.0), -70.0)

    def test_boundary_is_inclusive(self):
        """The comparison is >=, so a window exactly at threshold is kept."""
        window = window_with_power_db(-70.0)
        exact_threshold = compute_window_power_db(window)
        assert filter_window_by_power(window, exact_threshold)

    def test_just_below_threshold_dropped(self):
        """Complement to the boundary test: a hair under the line is excluded."""
        window = window_with_power_db(-70.0)
        just_above = compute_window_power_db(window) + 1e-6
        assert not filter_window_by_power(window, just_above)

    def test_result_is_boolean_valued(self):
        """
        The return is truth-valued but is numpy's np.bool_, not the built-in
        bool the `-> bool` annotation advertises. Callers that only branch on it
        are unaffected; anything doing an `is True` identity check or handing the
        value straight to a strict serializer would be. Asserted as-is here
        rather than worked around, so a future tightening of the return type
        shows up as a deliberate change.
        """
        result = filter_window_by_power(window_with_power_db(-50.0), -70.0)
        assert isinstance(result, (bool, np.bool_))
        assert bool(result) is True

    def test_zero_window_dropped_by_any_realistic_threshold(self):
        assert not filter_window_by_power(np.zeros((2, 64)), -100.0)

    def test_selection_over_a_mixed_batch(self):
        """End-to-end: only windows at or above the threshold survive."""
        powers = [-95.0, -80.0, -70.0, -60.0, -20.0]
        kept = [p for p in powers
                if filter_window_by_power(window_with_power_db(p), -70.0)]
        assert kept == [-70.0, -60.0, -20.0]
