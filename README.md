# VSDI Face Decoding

Python pipeline for VSDI face vs non-face decoding from MATLAB condition files.

## What This Project Does

This repository provides an end-to-end workflow for:
- converting raw `.mat` condition files to `.npy`
- frame-zero and blank-condition normalization
- dataset assembly (`X`, `y`) for face/non-face classification
- ROI-based feature extraction
- linear SVM decoding with grouped cross-validation
- sliding-window decoding over time
- visualization of accuracy, ROC/confusion, and weight maps

## Project Structure

```text
scripts/
  main.py                         # main decoding workflow (dataset -> ROI -> sliding window)
  optinal_preprocrssing.py        # optional preprocessing pipeline from raw .mat
  functions_scripts/
    preprocessing_functions.py     # split/normalize/build X,y/z-score
    feature_extraction.py          # ROI creation, window extraction, reshaping
    ml_cv.py                       # CV utilities + nested CV
    sliding_win.py                 # sliding-window decoding + plot
    ml_evaluation.py               # weight-map analysis utilities
    ml_plots.py                    # plotting helpers
    movie_function.py              # interactive weight-map movie

data/
  raw_mat/                         # input .mat files (not versioned)
  processed/                       # generated .npy outputs (not versioned)

figures/                           # saved figures
requirements.txt
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected Data Format

Input `.mat` files should contain keys like `cond1`, `cond2`, ...

Expected array shape through the pipeline:
- condition arrays: `(pixels, frames, trials)`
- decoded dataset: `X` as `(pixels, frames, trials)`, `y` as `(trials,)`

The current code assumes 100x100 spatial maps (`pixels = 10000`).

## Quick Start

### 1) Optional preprocessing from raw MATLAB files

Run:

```bash
python scripts/optinal_preprocrssing.py
```

This script is set up for:
1. split condition files
2. frame-zero normalization
3. blank normalization
4. ROI mask creation/saving

Note: parts of this script are currently commented out. Uncomment the preprocessing calls to generate outputs.

### 2) Main decoding analysis

Run:

```bash
python scripts/main.py
```

Current flow in `scripts/main.py`:
1. load processed face/non-face conditions with `build_X_y`
2. z-score data (`zscore_dataset_pixelwise_trials`)
3. load ROI mask (`data/processed/ROI_onlyV24_mask.npy`)
4. extract ROI + temporal window features
5. run sliding-window decoding (`sliding_window_decode_with_stats`)
6. plot accuracy curves and weight maps
7. show interactive weight movie

## Important Notes

- `scripts/main.py` and `scripts/optinal_preprocrssing.py` contain absolute Windows paths (for example `C:\project\vsdi-face-decoding\...`).
- If you run in a different location, update these paths to relative paths or your local absolute paths.
- Raw and processed data are intentionally excluded from version control.

## Outputs

Depending on what you run, outputs include:
- normalized `.npy` condition files under `data/processed/`
- ROI mask file (`ROI_onlyV24_mask.npy`)
- diagnostic figures (accuracy bars, confusion matrix, ROC, spatial weights, permutation/activation plots)
- interactive movie of windowed weight maps

## Reproducibility

For stable runs, keep:
- same trial splits (group definitions)
- same ROI mask
- same preprocessing settings (zero frames, blank condition)
- fixed model hyperparameters (`C`, `n_splits`, window size/range)
