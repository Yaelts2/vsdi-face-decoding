import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
print("Project root:", project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np

import matplotlib.pyplot as plt
from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import ml_plots as pl
from scripts.functions_scripts import sliding_win as sw
from scripts.functions_scripts.save_results import save_experiment
from scripts.functions_scripts import feature_extraction as fe
ourCmap = pre.green_gray_magenta()


## user must edit these parameters for each run! (see config section below)##
# =========================
# CONFIG
# =========================

# --- RT-aligned data (the data we actually want to decode) ---
face_file_rt = "RTaligned_110209a1_corrct.npy"
nonface_file_rt = "RTaligned_110209a5_corrct.npy"  # <-- EDIT to match your actual nonface RT file name
data_dir_rt = "data/processed/RTaligend/110209a/"

# --- Stimulus-aligned data for the SAME session/trials, used only to compute baseline ---
face_file_stim = "condsXn1_110209a.npy"
nonface_file_stim = "condsXn5_110209a.npy"  # <-- EDIT to match your actual nonface stim file name
data_dir_stim = "data/processed/condsXn/"

# Preprocessing:
# Baseline frames are indices into the STIM-ALIGNED data (true pre-stimulus window),
# NOT into the RT-aligned data. This is the whole point of the fix.
Baseline_frames_zscore = (1, 24)

# Feature extraction:
ROI_mask_path = "data/processed/ROI_110209_allA.npy"

# Model:
SEED = 42
model = lambda: cv.make_linear_svm(C=0.0001, max_iter=100000, random_state=SEED)
window_size = int(5)   # number of frames in sliding window
start_frame = int(1)   # first center frame to decode -- RT file is short (45 frames), check this fits
stop_frame = int(45)   # last center frame to decode -- EDIT based on actual RT-aligned frame count
step = int(1)
n_splits = int(10)

# save model results path:
results_root = "C:\\project\\vsdi-face-decoding\\results\\RTalignedModels"  # <-- EDIT to your actual results root path


# Expected core shapes:
# - X_trials: (pixels, frames, trials)
# - y_trials: (trials,)
# - groups: (samples,)              # samples = trials x frames


# ================== helper: per-trial baseline from stim-aligned data ==================

def zscore_RT_with_stim_baseline(X_stim, X_rt, baseline_frames):
    """
    Z-score RT-aligned trial data using per-trial, per-pixel baseline
    statistics computed from the SAME trial's stimulus-aligned data.

    This exists because the RT-aligned cut (30 frames before RT, 15 after)
    has no true pre-stimulus period of its own -- frames 1-24 of the RT cut
    can already contain real face/scramble-evoked and motor-related signal,
    and how much varies trial-to-trial with RT length. So baseline stats
    must come from the true pre-stimulus window of the ORIGINAL stimulus-
    aligned trial, then be applied to that same trial's RT-aligned cut.

    Parameters
    ----------
    X_stim : ndarray, shape (pixels, frames_stim, trials)
        Stimulus-onset aligned data for the SAME trials, SAME order as X_rt.
    X_rt : ndarray, shape (pixels, frames_rt, trials)
        RT-aligned data to be z-scored.
    baseline_frames : tuple (start, end)
        Frame indices into X_stim's frame axis (pre-stimulus window).

    Returns
    -------
    X_rt_z : ndarray, same shape as X_rt
        RT-aligned data z-scored using each trial's own stim-aligned baseline.
    mean : ndarray, shape (pixels, trials)
        Per-pixel, per-trial baseline mean.
    std : ndarray, shape (pixels, trials)
        Per-pixel, per-trial baseline std.
    """
    assert X_stim.shape[-1] == X_rt.shape[-1], (
        f"Trial count mismatch between stim-aligned ({X_stim.shape[-1]}) "
        f"and RT-aligned ({X_rt.shape[-1]}) data. Positional trial-order "
        f"matching assumes these line up 1:1 -- fix the mismatch before proceeding."
    )
    assert X_stim.shape[0] == X_rt.shape[0], (
        f"Pixel count mismatch between stim-aligned ({X_stim.shape[0]}) "
        f"and RT-aligned ({X_rt.shape[0]}) data."
    )

    b0, b1 = baseline_frames
    baseline = X_stim[:, b0:b1, :]          # (pixels, baseline_frames, trials)
    mean = np.nanmean(baseline, axis=1)     # (pixels, trials)
    std = np.nanstd(baseline, axis=1)       # (pixels, trials)

    n_zero_std = np.sum(std == 0)
    if n_zero_std > 0:
        print(f"WARNING: {n_zero_std} (pixel, trial) baseline std values are exactly 0 "
              f"-- these will become NaN after division. Investigate flat/dead pixels.")
    std_safe = np.where(std == 0, np.nan, std)

    X_rt_z = (X_rt - mean[:, None, :]) / std_safe[:, None, :]
    return X_rt_z, mean, std


# ================== preparing the data ==================

# --- Load stim-aligned data (baseline source only) ---
X_stim_trials, y_stim_trials, dataset_info_stim = pre.build_X_y(
    face_file=face_file_stim,
    nonface_file=nonface_file_stim,
    data_dir=data_dir_stim,
)
print(f"Stim-aligned dataset shape: {X_stim_trials.shape}, Labels shape: {y_stim_trials.shape}")

# --- Load RT-aligned data (the data to be decoded) ---
X_rt_trials, y_rt_trials, dataset_info_rt = pre.build_X_y(
    face_file=face_file_rt,
    nonface_file=nonface_file_rt,
    data_dir=data_dir_rt,
)
print(f"RT-aligned dataset shape: {X_rt_trials.shape}, Labels shape: {y_rt_trials.shape}")

# --- Sanity check: trial order/count assumed to match positionally ---
# Both files are correct-trials-only, so if the face/nonface trial counts
# also match, the resulting label vectors should be identical. This does NOT
# prove trial-by-trial identity, but it will catch the most likely failure
# mode (different trial counts / different face-nonface split point).
assert y_stim_trials.shape[0] == y_rt_trials.shape[0], (
    f"Trial count mismatch: stim-aligned has {y_stim_trials.shape[0]} trials, "
    f"RT-aligned has {y_rt_trials.shape[0]} trials. Positional matching is unsafe -- stopping."
)
if not np.array_equal(y_stim_trials, y_rt_trials):
    print("WARNING: label vectors from stim-aligned and RT-aligned data differ "
          "even though trial counts match. This suggests the face/nonface trial "
          "split boundary differs between the two files -- positional trial "
          "matching may be WRONG. Double-check before trusting these results.")
else:
    print("Sanity check passed: trial counts and face/nonface label order match "
          "between stim-aligned and RT-aligned data.")

y_trials = y_rt_trials  # use RT-aligned labels going forward

# --- Z-score RT-aligned data using each trial's own stim-aligned baseline ---
X_z, mean, std = zscore_RT_with_stim_baseline(X_stim_trials, X_rt_trials, Baseline_frames_zscore)
print("RT-aligned data z-scored using per-trial stimulus-aligned baseline.")
print("Baseline mean/std stats -- mean of means:", np.nanmean(mean), "mean of stds:", np.nanmean(std))


# feature extraction: window + ROI
## ROI selection
ROI_mask = np.load(ROI_mask_path).astype(bool)  # boolean mask in full image space (10000,) or (100,100)
pl.mimg(ROI_mask, xsize=100, ysize=100, low=0, high=1)
print(f"ROI mask loaded: {ROI_mask.shape}")
X_roi = X_z[ROI_mask, :, :]  # (roi_pixels x frames x trials)
print(f"ROI selected: {X_roi.shape}")


# ========= Run sliding window decoding ==========

results = sw.sliding_window_decode_with_stats(X_roi,
                                               y_trials,
                                               model,
                                               window_size,
                                               start_frame,
                                               stop_frame,
                                               step,
                                               n_splits)
results["zscore_std_pooled"] = std

# Checking the model
peak_idx = results["trial_acc_mean"].argmax()
clf_final = results["final_models"][peak_idx]
w = results["final_weights"][peak_idx]
b = results["final_intercept"][peak_idx]

start = start_frame + peak_idx * step
end = start + window_size
X_win = X_roi[:, start:end, :]
X_frames_check, y_frames_check, groups_check = fe.frames_as_samples(
    X_win, y_trials, trial_axis=2, frame_axis=1, pixel_axis=0
)

manual_score = X_frames_check @ w + b
sklearn_score = clf_final.decision_function(X_frames_check)
print("Max abs difference (scores):", np.max(np.abs(manual_score - sklearn_score)))

manual_pred = (manual_score > 0).astype(int)
sklearn_pred = clf_final.predict(X_frames_check)
print("Predictions match:", np.array_equal(manual_pred, sklearn_pred) or
      np.array_equal(1 - manual_pred, sklearn_pred))

print("Class order (classes_):", clf_final.classes_)


# --- Plot results ---
sw.plot_sliding_window_accuracy_with_std(res=results,
                                          chance=0.5,
                                          title="RT-Aligned Sliding Window Decoding (5-frame window)")

print("Peak frame accuracy:", results["frame_acc_mean"].max())
print("Peak trial accuracy:", results["trial_acc_mean"].max())
peak_idx = results["trial_acc_mean"].argmax()
print("Peak window center frame:", results["centers"][peak_idx])


# =========================
# SAVE RESULTS
# =========================

dataset_info_rt.update({
    "face_file": face_file_rt,
    "nonface_file": nonface_file_rt,
    "data_dir": data_dir_rt,
    "baseline_source": "stim_aligned_per_trial",
    "baseline_face_file_stim": face_file_stim,
    "baseline_nonface_file_stim": nonface_file_stim,
    "baseline_data_dir_stim": data_dir_stim,
    "zscore_baseline_frames_stim": Baseline_frames_zscore,
    "model": "linear SVM",
    "random_state": SEED,
    "window_size": window_size,
    "start_frame": start_frame,
    "stop_frame": stop_frame,
    "step": step,
    "n_splits": n_splits,
})

run_dir = save_experiment(results_root=results_root,
                           experiment="sliding_window_RTaligned",
                           experiment_tag=f"110209a15_RTaligned_frame{start_frame}-{stop_frame}__SVM_{n_splits}foldCV",
                           results=results,
                           ROI_mask_path=ROI_mask_path,
                           dataset_info=dataset_info_rt)

print(f"\nResults saved to: {run_dir}")

plt.show(block=True)