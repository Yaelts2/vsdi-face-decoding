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
from scripts.functions_scripts.errorTrials_Functions import load_error_trials
ourCmap = pre.green_gray_magenta()


# =========================
# CONFIG
# =========================

SESSION_TAG = "110209d4"

# Correct-trial (nonface) file:
nonface_file = "condsXn4_110209d.npy"
data_dir = "data/processed/condsXn/"

# Error-trial file:
error_mat_path = r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209d\errorTrialsData_110209d24.mat"
NONFACE_COND = 4  

# Preprocessing:
Baseline_frames_zscore = (1, 24)

# Feature extraction:
ROI_mask_path = "data/processed/v2_mask_110209.npy"  # confirm this is the right ROI mask for this session

# Class balance:
SEED = 42
ERROR_LABEL = 1
CORRECT_LABEL = 0

# Model:
model = lambda: cv.make_linear_svm(C=0.0001, max_iter=100000, random_state=SEED)
window_size = int(5)
start_frame = int(1)
stop_frame = int(100)
step = int(1)
n_splits = int(5)

results_root = "C:\\project\\vsdi-face-decoding\\results"


# ================== loading correct-trial (nonface) data ==================
correct_dir = Path(data_dir)
X_correct_all = np.load(correct_dir / nonface_file)  # (pixels, frames, n_correct_trials)
print(f"Loaded correct nonface trials: {X_correct_all.shape}")


# ================== loading error-trial data ==================
vsd_error_all, meta_error = load_error_trials(error_mat_path)
cond_mask = meta_error["condId"] == NONFACE_COND
X_error = vsd_error_all[:, :, cond_mask]
n_error = X_error.shape[2]
print(f"Loaded error trials, filtered to nonface (condId={NONFACE_COND}): {X_error.shape}")


# ================== matching N: subsample the larger group ==================
rng = np.random.default_rng(SEED)
n_correct_available = X_correct_all.shape[2]
n_error_available = X_error.shape[2]
n_match = min(n_correct_available, n_error_available)

if n_correct_available > n_match:
    correct_idx = rng.choice(n_correct_available, size=n_match, replace=False)
    correct_idx.sort()
    X_correct = X_correct_all[:, :, correct_idx]
else:
    correct_idx = np.arange(n_correct_available)
    X_correct = X_correct_all

if n_error_available > n_match:
    error_idx = rng.choice(n_error_available, size=n_match, replace=False)
    error_idx.sort()
    X_error = X_error[:, :, error_idx]
else:
    error_idx = np.arange(n_error_available)

n_error = n_match  # keep downstream variable name consistent
print(f"Matched N: correct={X_correct.shape[2]}, error={X_error.shape[2]} (target n={n_match})")


# ================== save matched subsamples in build_X_y-compatible format ==================
# This lets the STANDARD opener script (sr.load_data_from_config) work
# unmodified -- no need for a separate correctVsError-specific opener.
# build_X_y labels file #1 as 1, file #2 as 0, which matches
# ERROR_LABEL=1 / CORRECT_LABEL=0 exactly if error goes in the "face_file"
# slot and correct goes in the "nonface_file" slot.

saved_data_dir = Path(data_dir) / "correctVsError_subsamples"
saved_data_dir.mkdir(parents=True, exist_ok=True)

error_save_name = f"{SESSION_TAG}_error_subsample.npy"
correct_save_name = f"{SESSION_TAG}_correct_subsample.npy"

np.save(saved_data_dir / error_save_name, X_error)
np.save(saved_data_dir / correct_save_name, X_correct)
print(f"Saved matched subsamples to: {saved_data_dir}")
print(f"  error_subsample -> {error_save_name}")
print(f"  correct_subsample -> {correct_save_name}")

# ================== stacking + labeling ==================
# Order: error trials first, then correct trials (mirrors build_X_y's
# "first half = positive label" convention, with ERROR_LABEL=1 here)
X_trials = np.concatenate([X_error, X_correct], axis=2)  # (pixels, frames, 2*n_error)
y_trials = np.array([ERROR_LABEL] * n_error + [CORRECT_LABEL] * n_error)

dataset_info = {
    # --- standard keys, for compatibility with sr.load_data_from_config ---
    "face_file": error_save_name,
    "nonface_file": correct_save_name,
    "data_dir": str(saved_data_dir),

    # --- correctVsError-specific keys, kept for reference/documentation ---
    "n_trials_per_class": n_match,
    "session_tag": SESSION_TAG,
    "original_nonface_file": nonface_file,   # renamed from "nonface_file" to avoid clashing with the standard key above
    "error_mat_path": error_mat_path,
    "nonface_cond": NONFACE_COND,
    "error_label": ERROR_LABEL,
    "correct_label": CORRECT_LABEL,
    "subsample_seed": SEED,
    "correct_trial_subsample_idx": correct_idx.tolist(),
    "error_trial_subsample_idx": error_idx.tolist(),
    "n_correct_available": n_correct_available,
    "n_error_available": n_error_available,
}
print(f"Final: X shape: {X_trials.shape}, y: {y_trials}")


# ================== z-score (fresh, across combined correct+error set) ==================
X_z, mean, std = pre.zscore_dataset_pixelwise_trials(X_trials, Baseline_frames_zscore)
b = X_z[:, Baseline_frames_zscore[0]:Baseline_frames_zscore[1], :]
print(np.nanmean(b), np.nanstd(b))  # should be close to 0 and 1 (global check)
print("Data z-scored across combined correct+error set.")


# ================== feature extraction: window + ROI ==================
ROI_mask = np.load(ROI_mask_path).astype(bool)
pl.mimg(ROI_mask, xsize=100, ysize=100, low=0, high=1)
print(f"ROI mask loaded: {ROI_mask.shape}")
X_roi = X_z[ROI_mask, :, :]
print(f"ROI selected: {X_roi.shape}")


# ========= Run sliding window decoding: correct vs. error =========

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
b_ = results["final_intercept"][peak_idx]

start = start_frame + peak_idx * step
end = start + window_size
X_win = X_roi[:, start:end, :]
X_frames_check, y_frames_check, groups_check = fe.frames_as_samples(
    X_win, y_trials, trial_axis=2, frame_axis=1, pixel_axis=0
)

manual_score = X_frames_check @ w + b_
sklearn_score = clf_final.decision_function(X_frames_check)
print("Max abs difference (scores):", np.max(np.abs(manual_score - sklearn_score)))

manual_pred = (manual_score > 0).astype(int)
sklearn_pred = clf_final.predict(X_frames_check)
print("Predictions match:", np.array_equal(manual_pred, sklearn_pred) or
    np.array_equal(1 - manual_pred, sklearn_pred))

print("Class order (classes_):", clf_final.classes_)
print(f"Number of fold models at peak window: {len(results['fold_models_all'][peak_idx])}")


# --- Plot results ---
sw.plot_sliding_window_accuracy_with_std(res=results,
                                        chance=0.5,
                                        title=f"[{SESSION_TAG}] Correct vs. Error decoding (nonface only)")

print("Peak frame accuracy:", results["frame_acc_mean"].max())
print("Peak trial accuracy:", results["trial_acc_mean"].max())
print("Peak window center frame:", results["centers"][peak_idx])


# =========================
# SAVE RESULTS
# =========================

dataset_info.update({
    "zscore_baseline_frames": Baseline_frames_zscore,
    "model": "linear SVM",
    "random_state": SEED,
    "window_size": window_size,
    "start_frame": start_frame,
    "stop_frame": stop_frame,
    "step": step,
    "n_splits": n_splits,
})

run_dir = save_experiment(results_root=results_root,
                        experiment="sliding_window",
                        experiment_tag=f"{SESSION_TAG}_v2_correctVsError_nonface_frame{start_frame}-{stop_frame}__SVM_{n_splits}foldCV",
                        results=results,
                        ROI_mask_path=ROI_mask_path,
                        dataset_info=dataset_info)

print(f"\nResults saved to: {run_dir}")