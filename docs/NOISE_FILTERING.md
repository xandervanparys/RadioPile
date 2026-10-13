# Noise Filtering

This document explains how to use noise filtering in the RF data preprocessing pipeline.

## Overview

Noise filtering allows you to automatically remove windows that contain only noise (below a certain power threshold) from your datasets. This is useful for:

- Removing low-quality windows that don't contain useful signal information
- Improving dataset quality for machine learning
- Reducing dataset size by filtering out noise-only windows

The noise floor is typically a characteristic of the recording equipment and environment, so it is detected at the **file level**, not per window.

## Configuration

Noise filtering is configured in your dataset's `config.py` file using the `NOISE_POWER_THRESHOLD_DB` parameter.

### Option 1: Manual Threshold (Recommended for Production)

Set a fixed threshold in dB. Windows with power below this threshold will be filtered out.

```python
NOISE_POWER_THRESHOLD_DB = -70.0  # Filter windows below -70 dB
```

### Option 2: Auto-Detection

Automatically detect the noise floor for each file using one of several methods.

```python
NOISE_POWER_THRESHOLD_DB = "auto"
NOISE_FLOOR_DETECTION_METHOD = "percentile"  # or "histogram" or "statistical"
NOISE_FLOOR_DETECTION_PARAMS = {
    "percentile": 1.0,  # Floor = 1st percentile of window power (the v1 setting)
    "margin_db": 6.0,   # Threshold = floor + 6 dB
}
NOISE_FLOOR_MAX_SAMPLES = 10000  # Max windows to sample per file
```

### Option 3: Disable Filtering

```python
NOISE_POWER_THRESHOLD_DB = None
```

## Detection Methods

Three automatic detection methods are available: `"percentile"`, `"histogram"` and
`"statistical"`. What each one computes, and the values the v1 sources use, are in
[`DESIGN.md`](DESIGN.md) §2.6.

## Visual Inspection

**Recommended: Use the Jupyter notebook** (`notebooks/inspect_noise_floor.ipynb`) for interactive exploration. The notebook provides:
- Interactive visualization of power distributions
- Side-by-side comparison of detection methods
- Filtering impact analysis
- Easy configuration snippet generation

Alternatively, you can use the command-line script:

```bash
python scripts/inspect_noise_floor.py <file_path> --compare
```

A `.bin` file carries no format information, so both tools must be told its on-disk
dtype: `--source-dtype` for the script (default `complex64`) and `SOURCE_DTYPE` in the
notebook's configuration cell. `--save-plot FILE` writes the figure instead of showing
it.

Both tools show a histogram of window powers and compare the detection methods.

Example output:
```
Comparing detection methods:
--------------------------------------------------
Results:
  percentile (5%)    :  -72.45 dB
  percentile (10%)   :  -70.23 dB
  histogram          :  -71.89 dB
  statistical        :  -73.12 dB
```

Without `--compare` the script prints one estimate per method and their average as a
suggested threshold.

## Integration

Noise filtering is automatically applied during dataset building:

1. If `NOISE_POWER_THRESHOLD_DB` is a number: All windows below this threshold are filtered
2. If `NOISE_POWER_THRESHOLD_DB == "auto"`: Noise floor is detected per file, then windows below the detected threshold are filtered
3. If `NOISE_POWER_THRESHOLD_DB == None`: No filtering is applied

Filtering happens **before normalization**, so the threshold is applied to the raw window power.

## Example Workflow

1. **Inspect a sample file** to understand the power distribution:
   - Open `notebooks/inspect_noise_floor.ipynb` in Jupyter
   - Update the `file_path` variable to point to your sample file
   - Run all cells to see visualizations and get suggested thresholds

   Or use the command-line script:
   ```bash
   python scripts/inspect_noise_floor.py data/your_dataset/raw/sample_file.bin --source-dtype complex64 --compare
   ```

2. **Choose a threshold** based on the visualization and method comparison

3. **Configure your dataset** in `config.py`:
   ```python
   # Option A: Use the suggested threshold manually
   NOISE_POWER_THRESHOLD_DB = -71.7

   # Option B: Use auto-detection
   NOISE_POWER_THRESHOLD_DB = "auto"
   NOISE_FLOOR_DETECTION_METHOD = "percentile"
   NOISE_FLOOR_DETECTION_PARAMS = {"percentile": 1.0, "margin_db": 6.0}
   ```

4. **Build your dataset** - filtering will be applied automatically:
   ```bash
   python run.py your_dataset
   ```

The build process will show filtering statistics:
```
Auto-detected noise floor: -72.45 dB
Added margin: 6.0 dB → threshold: -66.45 dB
Filtered 1234/10000 windows (12.3%)
```

## Notes

- The noise floor is detected per **file**, not per window, because it is a characteristic of the recording setup
- The threshold is in **dB** (decibels), computed as `10*log10(mean(I² + Q²))`
- Filtering happens **before normalization**, so thresholds are relative to raw signal power
- Auto-detection samples up to `NOISE_FLOOR_MAX_SAMPLES` windows per file for efficiency
- For very large files, auto-detection may take a moment but is still efficient
