from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
print("Project root:", project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import matplotlib.pyplot as plt
import json, re
from scipy.io import savemat
from functions_scripts import preprocessing_functions as pre
from functions_scripts import Weights_Evaluation as ev
from functions_scripts import ml_plots as pl
from functions_scripts import save_results as sr
from functions_scripts import sliding_win as sw
from functions_scripts import movie_function as mf
from functions_scripts.errorTrials_Functions import load_error_trials

ourCmap = pre.green_gray_magenta()
###############
# Analysis for correct-vs-error sliding window experiment #
###############

## user must edit these parameters for each run!##
### which model to load and plot results from

results_root = Path(r"C:\project\vsdi-face-decoding\results")
model_root = results_root / "sliding_window__110209d4_correctVsError_nonface_frame1-100__SVM_5foldCV____2026-08-04_14-23-52"  # <-- update to your actual run folder
print("model_root:", model_root)


### Load the experiment results and config
config_sliding_window, results_sliding_window, ROI_mask_path = sr.load_experiment(str(model_root))
print("Experiment config:", config_sliding_window)

### reconstruct the exact correct+error trial set used for this run
# NOTE: intentionally NOT using sr.load_data_from_config here -- that path
# assumes the standard face_file/nonface_file build_X_y layout, which doesn't
# match this experiment's config keys (nonface_file=correct-trial source,
# error_mat_path, nonface_cond, subsample indices). Reconstructing manually
# from the saved subsample indices instead, so we reload the SAME trials
# the model was trained on, not just a same-sized fresh sample.

data_dir = Path(config_sliding_window.get("data_dir", "data/processed/condsXn/"))
nonface_file = config_sliding_window["nonface_file"]
error_mat_path = config_sliding_window["error_mat_path"]
nonface_cond = config_sliding_window["nonface_cond"]
error_label = config_sliding_window["error_label"]
correct_label = config_sliding_window["correct_label"]
correct_idx = np.asarray(config_sliding_window["correct_trial_subsample_idx"])
error_idx = np.asarray(config_sliding_window["error_trial_subsample_idx"])

X_correct_all = np.load(data_dir / nonface_file)
X_correct = X_correct_all[:, :, correct_idx]

vsd_error_all, meta_error = load_error_trials(error_mat_path)
cond_mask = meta_error["condId"] == nonface_cond
X_error_all = vsd_error_all[:, :, cond_mask]
X_error = X_error_all[:, :, error_idx]

X_trials = np.concatenate([X_error, X_correct], axis=2)
n_per_class = X_error.shape[2]
y_trials = np.array([error_label] * n_per_class + [correct_label] * n_per_class)
print("Reconstructed X_trials, y_trials:", X_trials.shape, y_trials.shape)

# re-apply the SAME fresh z-score used during training (combined correct+error set)
X_z, mean, std = pre.zscore_dataset_pixelwise_trials(
    X_trials, config_sliding_window["zscore_baseline_frames"]
)

#load the ROI mask and apply it to the data
ROI_mask = np.load(ROI_mask_path)
print("ROI_mask shape:", ROI_mask.shape)
X_ROI = X_z[ROI_mask, :, :]
print("X_ROI shape:", X_ROI.shape)


# =========================
# PLOTS
# =========================

W_img = ev.window_weights_to_pixel_time_matrix(results_sliding_window["w_mean_windows"], ROI_mask, pixels=100)
print(W_img.shape)  # (pixels, frames) where frames correspond to window centers

pl.mimg(W_img, xsize=100, ysize=100, low=-0.0002, high=0.0002, frames=results_sliding_window["centers"], colormap=ourCmap)
frame_ids = np.arange(0, W_img.shape[1])
weights = W_img
X_avg, labels_ms, bin_frames = pl.avg_consecutive_frames_with_ms_labels(X=weights, frame_ids=frame_ids, avg_n=3, dt_ms=10, zero_frame=27
)

fig, axes_flat = pl.mimg(X_avg, xsize=100, ysize=100, low=-0.001, high=0.001, colormap=ourCmap, frames=labels_ms+10)
plt.show()


#mimg of positive and negative weight masks across windows
pos_masks, neg_masks = ev.extract_extreme_weight_masks_sliding(W_img, ROI_mask, frac=0.20)
pl.mimg(pos_masks, xsize=100, ysize=100, low='auto', high=None, frames=results_sliding_window["centers"])
pl.mimg(neg_masks, xsize=100, ysize=100, low=0, high=1, frames=results_sliding_window["centers"])


sw.plot_sliding_window_accuracy_with_std(
    results_sliding_window,
    zero_frame=27,          # frame 27 = 0 ms
    frame_duration_ms=10,   # 10 ms per frame
)



a = 1



# =========================
# EXPORT FOR MATLAB (openModel.m) — plotting data + metadata
# =========================
R   = results_sliding_window
cfg = config_sliding_window
name = model_root.name

# --- parse date + session + conditions from the folder name ---
m = re.search(r'_(\d+)([A-Za-z])(\d+)_', name)
if m:
    sess_code  = m.group(1)
    session    = m.group(2)
    conditions = ",".join(m.group(3))
else:
    sess_code, session, conditions = "", "", ""
    print("WARNING: could not parse session/conditions from name:", name)

save_dir = Path(r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\saveDataModels\errorVsCorrect")
save_dir.mkdir(parents=True, exist_ok=True)
out_path = save_dir / (name + ".mat")

export = {
    "model_id":           name,
    "mat_path":           str(out_path),
    "date":               sess_code,
    "session":            session,
    "conditions":         conditions,
    "mask":               Path(ROI_mask_path).stem,
    "window_size":        np.float64(cfg["window_size"]),
    "window_step":        np.float64(cfg["step"]),
    "n_folds":            np.float64(cfg["n_splits"]),
    "n_trials_per_class": np.float64(cfg["n_trials_per_class"]),
    "config_json":        json.dumps(cfg, default=str),

    "Weights_acrossT":    np.asarray(W_img, dtype=np.float64),
    "centers":            np.asarray(R["centers"], dtype=np.float64).ravel(),
    "zero_frame":         np.float64(27),
    "frame_duration_ms":  np.float64(10),
    "frame_acc_mean":     np.asarray(R["frame_acc_mean"], dtype=np.float64).ravel(),
    "trial_acc_mean":     np.asarray(R["trial_acc_mean"], dtype=np.float64).ravel(),
    "frame_acc_std":      np.asarray(R.get("frame_acc_std", np.zeros_like(R["frame_acc_mean"])), dtype=np.float64).ravel(),
    "trial_acc_std":      np.asarray(R.get("trial_acc_std", np.zeros_like(R["trial_acc_mean"])), dtype=np.float64).ravel(),
    "trial_acc_folds":    np.asarray(R["fold_trial_acc"], dtype=np.float64),
}

savemat(str(out_path), export)
print("saved:", out_path)
print(f"  date={sess_code} session={session} conditions={conditions} "
    f"win={cfg['window_size']} step={cfg['step']} folds={cfg['n_splits']} ntrials={cfg['n_trials_per_class']}")