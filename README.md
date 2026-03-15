# VSDI Face Decoding

Pipeline for decoding face vs non-face VSDI activity using fixed-window and sliding-window linear SVM analyses, with permutation testing and post-hoc visualization.

## Project Structure

```text
vsdi-face-decoding/
  data/
    raw_mat/                       # original MATLAB files (conds_*.mat)
    processed/                     # processed .npy files + ROI masks
  figures/                         # exported plots
  results/                         # saved run folders (config + result files)
  scripts/
    optional_preprocessing.py      # optional raw .mat -> normalized .npy + ROI workflow

    run_experiment/
      run_fixed_window.py          # fixed temporal window decoding + nested CV
      run_sliding_window.py        # sliding-window decoding across time
      run_permutation_test.py      # permutation test for a fixed-window saved run
      run_permutation_test_slidingwin   # permutation test for a sliding-window saved run

    open_fixed_win_model.py        # inspect/plot fixed-window saved run
    open_sliding_win_model.py      # inspect/plot sliding-window saved run
    open_permutation_fixed_win.py  # inspect/plot fixed-window permutation results
    open_permutation_sliding.py    # inspect/plot sliding-window permutation results

    functions_scripts/
      preprocessing_functions.py   # data loading + normalization helpers
      feature_extraction.py        # ROI/window/frame feature builders
      ml_cv.py                     # SVM + CV/nested CV
      sliding_win.py               # sliding-window training/evaluation
      model_control.py             # permutation/statistics utilities
      ml_evaluation.py             # weight/activation analyses
      ml_plots.py                  # plotting helpers
      movie_function.py            # weight movie visualization
      save_results.py              # save/load run artifacts

  README.md
  requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Data Expectations

Input data is expected as `.npy` condition arrays (typically generated from `.mat` files), with internal shape conventions used across the code:
- trial-level arrays: `(pixels, frames, trials)`
- labels: `(trials,)`
- frame-as-sample arrays (for model fitting): `(samples, features)`

Most scripts assume 100x100 maps (`pixels = 10000`) and a saved ROI mask in `data/processed/`.

## Saved Models and Results
This project generates several types of outputs, including fixed-window classifications, sliding-window decodings, and their respective permutation tests.

Dataset 1 (030209a)
Fixed Window (Frames 32-40):
"fixed_window__frame32-40__SVM_10fold__2026-03-02_14-06-30"

Dataset 2 (110209a)
Early Fixed Window (Frames 32-40): 
"fixed_window__frame32-40__SVM_10fold__2026-02-24_15-48-01"

Late Fixed Window (Frames 47-55): 
"fixed_window__frame47-55__SVM_10fold__2026-03-02_14-55-16"

Permutation (Early Window): 
"fixed_window__frame32-40__SVM_10fold__2026-02-18_17-43-53__permutation__perm100__seed42__2026-02-20_01-59-40"

Sliding Window (Frames 0-100): 
"sliding_window__frame0-100__SVM_5foldCV__2026-02-25_15-08-54"

Permutation (Sliding Window):
"slidingwindow__perm1000__seed42__2026-02-26_17-34-13"

[!IMPORTANT]
How to Load Results:
To analyze or plot one of these runs, copy the specific folder name above and paste it into the RUN_DIR variable in the appropriate script:
For fixed-window models: Use open_fixed_win_model.py.
For sliding-window decodings: Use open_sliding_win_model.py.
For permutation tests: Use open_permutation_results.py.



## Main Workflows

### 1) Optional preprocessing from raw MATLAB files

```bash
python scripts/optional_preprocessing.py
```

Use this script to:
- split raw condition files
- apply frame-zero normalization
- apply blank-condition normalization
- create/save an ROI mask

Note: several preprocessing calls are currently commented in the script; enable the sections you want to run.

### 2) Train a fixed-window model

```bash
python -m scripts.run_experiment.run_fixed_window
```

What it does:
- builds `X_trials, y_trials`
- z-scores by baseline frames
- applies ROI mask and selected temporal window
- converts to frames-as-samples
- runs nested CV (`GroupKFold`) to select/evaluate `C`
- saves run under `results/fixed_window__...`

### 3) Train a sliding-window model

```bash
python -m scripts.run_experiment.run_sliding_window
```

What it does:
- same preprocessing/ROI setup
- decodes across moving windows
- stores frame-level and trial-level accuracy curves
- saves run under `results/sliding_window__...`

### 4) Run permutation tests

Fixed-window permutation test:

```bash
python -m scripts.run_experiment.run_permutation_test
```

Sliding-window permutation test:

```bash
python scripts/run_experiment/run_permutation_test_slidingwin
```

Both scripts load an existing saved run from `results/`, rebuild matching features, run label-shuffle tests, and save outputs under:
- `results/permutation_test/`
- `results/permutation_test_sliding_window/`

### 5) Open and visualize saved runs

```bash
python scripts/open_fixed_win_model.py
python scripts/open_sliding_win_model.py
python scripts/open_permutation_fixed_win.py
python scripts/open_permutation_sliding.py
```

These scripts load saved outputs and generate figures such as:
- fold-wise accuracy summaries
- confusion matrices / ROC
- weight maps and positive/negative mask activations
- sliding-window accuracy curves
- permutation null vs real-performance plots

## Important Configuration Notes

Most run/open scripts contain user-editable config blocks at the top (file names, ROI path, windows, CV setup, run directory to load).

Several scripts currently include absolute Windows paths like:
- `C:\project\vsdi-face-decoding\results`
- `C:\project\vsdi-face-decoding\data\...`

If your project is in a different location, update these paths before running.

## Output Format

A typical saved run directory contains:
- `config.json` - run configuration and dataset metadata
- `result.npz` - model outputs/metrics
- `ROI_mask_path.npz` - reference to ROI mask used

Permutation run directories contain:
- `config.json`
- `perm_result.npz`

## Reproducibility

For consistent results across runs, keep fixed:
- input condition files
- preprocessing settings (baseline/blank/normalization)
- ROI mask
- CV definitions (`GroupKFold` splits)
- model hyperparameters (`C`, window definitions, seeds)
