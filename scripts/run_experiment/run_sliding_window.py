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

# Data files:
face_file= "condsXn2_030209f.npy"
nonface_file= "condsXn4_030209f.npy"
data_dir= "data/processed/condsXn/"

# Preprocessing:
Baseline_frames_zscore= (1, 24)

# Feature extraction:
ROI_mask_path= "data/processed/ROI_030209all_mask.npy"

# Model :
SEED = 42
model=lambda: cv.make_linear_svm(C=0.0001, max_iter=100000, random_state=SEED)
window_size=int(5)  # number of frames in sliding window
start_frame=int(1) # first center frame to decode (e.g. 15 means window covers frames 13-17)
stop_frame=int(100) # last center frame to decode (e.g. 125 means window covers frames 123-127, but if stop_frame=126 it would be last center frame 124 with window covering 122-126)
step=int(1) # step size to move window (e.g. 1 means decode every center frame, 5 means decode every 5th center frame)
n_splits=int(10) # kfold for each window

# save model results path:
results_root = "C:\\project\\vsdi-face-decoding\\results"


# Expected core shapes:
# - X_trials: (pixels, frames, trials)
# - y_trials: (trials,)
# - groups: (samples,)              # samples = trials x frames




# ================== preparing the data ==================
#Build dataset (X,y)
# X shape must be: (trials, pixels, frames)
X_trials, y_trials,dataset_info = pre.build_X_y(face_file=face_file,
                                                nonface_file=nonface_file,
                                                data_dir=data_dir)
print(f"Dataset shape: {X_trials.shape}, Labels shape: {y_trials.shape}")

#Z score across all trials
X_z,mean,std=pre.zscore_dataset_pixelwise_trials(X_trials,Baseline_frames_zscore)
b = X_z[:, Baseline_frames_zscore[0]:Baseline_frames_zscore[1], :]
print(np.nanmean(b), np.nanstd(b))  # should be close to 0  and 1 (global check)
print("Data z-scored across all trials.")
#a=np.nanmean(X_z, axis=(2))
#binned, fig, axes, cid = pl.plot_superpixel_traces(a[:,25:125], xs=100, ys=100, nsubplots=15)
#plt.show()


#feature extraction: window + ROI
##ROI selection
ROI_mask = np.load(ROI_mask_path).astype(bool) # boolean mask in full image space (10000,) or (100,100)
pl.mimg(ROI_mask, xsize=100, ysize=100, low=0, high=1)
print(f"ROI mask loaded: {ROI_mask.shape}")
X_roi=X_z[ROI_mask,:,:] # (roi_pixels x frames x trials)
print(f"ROI selected: {X_roi.shape}")



# ========= Run sliding window decoding ==========

results = sw.sliding_window_decode_with_stats(X_roi,      # shape: (8518, 256, 56)
                                            y_trials,          # shape: (56,)
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

# Recompute X_frames for that same window (same slicing as inside the function)
start = start_frame + peak_idx * step
end = start + window_size
X_win = X_roi[:, start:end, :]
X_frames_check, y_frames_check, groups_check = fe.frames_as_samples(
    X_win, y_trials, trial_axis=2, frame_axis=1, pixel_axis=0
)

# Manual decision score vs. sklearn's own decision_function
manual_score = X_frames_check @ w + b
sklearn_score = clf_final.decision_function(X_frames_check)

print("Max abs difference (scores):", np.max(np.abs(manual_score - sklearn_score)))

# Predicted labels should match exactly
manual_pred = (manual_score > 0).astype(int)  # careful: see class order check below
sklearn_pred = clf_final.predict(X_frames_check)
print("Predictions match:", np.array_equal(manual_pred, sklearn_pred) or 
    np.array_equal(1 - manual_pred, sklearn_pred))

print("Class order (classes_):", clf_final.classes_)
print("Number of fold models at window 32:", len(results["fold_models_all"][32]))





# --- Plot results ---
sw.plot_sliding_window_accuracy_with_std(res=results,
                                        chance=0.5,
                                        title="Sliding Window Decoding (5-frame window)")

print("Peak frame accuracy:",
    results["frame_acc_mean"].max())

print("Peak trial accuracy:",
    results["trial_acc_mean"].max())

peak_idx = results["trial_acc_mean"].argmax()
print("Peak window center frame:",
    results["centers"][peak_idx])


# =========================
# SAVE RESULTS
# =========================

dataset_info.update({"face_file": face_file,
                    "nonface_file": nonface_file,
                    "data_dir": data_dir,
                    "zscore_baseline_frames": Baseline_frames_zscore,
                    "model": "linear SVM",
                    "random_state": SEED,
                    "window_size": window_size,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                    "step": step,
                    "n_splits": n_splits})

run_dir = save_experiment(results_root=results_root,
                        experiment="sliding_window",
                        experiment_tag=f"030209f24_frame{start_frame}-{stop_frame}__SVM_{n_splits}foldCV",
                        results=results,
                        ROI_mask_path=ROI_mask_path,
                        dataset_info=dataset_info)

print(f"\nResults saved to: {run_dir}")
