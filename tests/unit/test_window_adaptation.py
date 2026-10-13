"""
Tests for datasets/aggregation/build_dataset.py :: adapt_window_size

Covers zero-padding, start-cropping, and same-size passthrough,
plus the metadata flags that track what was done to each window.
"""

import numpy as np
import pytest

from datasets.aggregation.build_dataset import adapt_window_size


def make_window(samples: int, fill: float = 1.0) -> np.ndarray:
    """Create a synthetic (2, samples) float32 window filled with a constant."""
    return np.full((2, samples), fill, dtype=np.float32)


# ── Zero padding (source < target) ───────────────────────────────────────────

class TestZeroPadding:

    def test_output_shape(self):
        """(2, 128) → (2, 500) after zero padding."""
        window = make_window(128)
        adapted, _ = adapt_window_size(window, 128, 500, "zero", "start")
        assert adapted.shape == (2, 500)

    def test_original_samples_preserved(self):
        """The first source_size samples must equal the original window."""
        window = make_window(128, fill=2.5)
        adapted, _ = adapt_window_size(window, 128, 500, "zero", "start")
        np.testing.assert_array_equal(adapted[:, :128], window)

    def test_padded_region_is_zero(self):
        """Samples beyond source_size must be exactly 0.0."""
        window = make_window(128, fill=1.0)
        adapted, _ = adapt_window_size(window, 128, 500, "zero", "start")
        np.testing.assert_array_equal(adapted[:, 128:], 0.0)

    def test_was_padded_true(self):
        window = make_window(128)
        _, meta = adapt_window_size(window, 128, 500, "zero", "start")
        assert meta["was_padded"] is True

    def test_was_cropped_false(self):
        window = make_window(128)
        _, meta = adapt_window_size(window, 128, 500, "zero", "start")
        assert meta["was_cropped"] is False

    def test_padding_strategy_recorded(self):
        window = make_window(128)
        _, meta = adapt_window_size(window, 128, 500, "zero", "start")
        assert meta["padding_strategy"] == "zero"

    def test_original_window_size_recorded(self):
        window = make_window(128)
        _, meta = adapt_window_size(window, 128, 500, "zero", "start")
        assert meta["original_window_size"] == 128


# ── Start-crop (source > target) ─────────────────────────────────────────────

class TestStartCrop:

    def test_output_shape(self):
        """(2, 1000) → (2, 500) after start-crop."""
        window = make_window(1000)
        adapted, _ = adapt_window_size(window, 1000, 500, "zero", "start")
        assert adapted.shape == (2, 500)

    def test_content_is_first_n_samples(self):
        """Start-crop must keep the first target_size samples."""
        window = np.arange(1000, dtype=np.float32)[np.newaxis].repeat(2, axis=0)
        adapted, _ = adapt_window_size(window, 1000, 500, "zero", "start")
        np.testing.assert_array_equal(adapted, window[:, :500])

    def test_was_cropped_true(self):
        window = make_window(1000)
        _, meta = adapt_window_size(window, 1000, 500, "zero", "start")
        assert meta["was_cropped"] is True

    def test_was_padded_false(self):
        window = make_window(1000)
        _, meta = adapt_window_size(window, 1000, 500, "zero", "start")
        assert meta["was_padded"] is False

    def test_crop_policy_recorded(self):
        window = make_window(1000)
        _, meta = adapt_window_size(window, 1000, 500, "zero", "start")
        assert meta["crop_policy"] == "start"

    def test_original_window_size_recorded(self):
        window = make_window(1000)
        _, meta = adapt_window_size(window, 1000, 500, "zero", "start")
        assert meta["original_window_size"] == 1000


# ── Same size passthrough ─────────────────────────────────────────────────────

class TestSameSizePassthrough:

    def test_shape_unchanged(self):
        window = make_window(500)
        adapted, _ = adapt_window_size(window, 500, 500, "zero", "start")
        assert adapted.shape == (2, 500)

    def test_content_unchanged(self):
        window = make_window(500, fill=3.14)
        adapted, _ = adapt_window_size(window, 500, 500, "zero", "start")
        np.testing.assert_array_equal(adapted, window)

    def test_was_padded_false(self):
        window = make_window(500)
        _, meta = adapt_window_size(window, 500, 500, "zero", "start")
        assert meta["was_padded"] is False

    def test_was_cropped_false(self):
        window = make_window(500)
        _, meta = adapt_window_size(window, 500, 500, "zero", "start")
        assert meta["was_cropped"] is False


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_unknown_padding_strategy_raises(self):
        window = make_window(128)
        with pytest.raises(ValueError, match="Unknown padding strategy"):
            adapt_window_size(window, 128, 500, "invalid_strategy", "start")

    def test_unknown_crop_policy_raises(self):
        window = make_window(1000)
        with pytest.raises(ValueError, match="Unknown crop policy"):
            adapt_window_size(window, 1000, 500, "zero", "invalid_policy")
