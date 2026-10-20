# run_permutation_test_slidingwin.py
# Sliding-window permutation test for ONE model.
# Same as before + saves the weights of every shuffled model (for the overlap null in MATLAB).
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import json
import numpy as np
from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import model_control as mc
from scripts.functions_scripts import save_results as sr

# =========================
# CONFIG  (edit per dataset)
# =========================
RESULTS_ROOT = Path(r"C:\project\vsdi-face-decoding\results")
RUN_DIR = RESULTS_ROOT / "sliding_window__110209a24_frame1-100__SVM_10foldCV____2026-06-14_12-01-00"  # <- EDIT

N_PERMUTATIONS = 100
PERM_SEED = 42
SVM_SEED = 42

# keep the null weights only for windows in this time range (ms from stimulus onset)
SAVE_T_MS = (0, 300)
ZERO_FRAME = 27
FRAME_MS = 10

PERM_ROOT = RESULTS_ROOT / "permutation_test_sliding_window"

# =========================
# 1) Load the real model
# =========================
config, real_results, ROI_mask_path = sr.load_experiment(RUN_DIR)
print("Loaded run:", RUN_DIR.name)

face_file = config["face_file"]
nonface_file = config["nonface_file"]
data_dir = config["data_dir"]
baseline = tuple(config["zscore_baseline_frames"])

window_size = int(config.get("window_size", 5))
start_frame = int(config.get("start_frame", 1))
stop_frame = int(config.get("stop_frame", 100))
step = int(config.get("step", 1))
n_splits = int(config.get("n_splits", 10))

make_estimator = lambda: cv.make_linear_svm(C=0.0001, max_iter=100000, random_state=SVM_SEED)

# =========================
# 2) Rebuild the data exactly as in training
# =========================
X_trials, y_trials, _ = pre.build_X_y(face_file=face_file, nonface_file=nonface_file, data_dir=data_dir)
X_z, _mean, _std = pre.zscore_dataset_pixelwise_trials(X_trials, baseline_frames=baseline)
ROI_mask = np.asarray(np.load(ROI_mask_path), dtype=bool).ravel()
X_roi = X_z[ROI_mask, :, :]
print("X_roi:", X_roi.shape, " y:", y_trials.shape)

# =========================
# 3) Windows whose null weights are saved
# =========================
centers = np.asarray(real_results["centers"], dtype=int)
t_ms = (centers + 1 - ZERO_FRAME) * FRAME_MS
save_centers = centers[(t_ms >= SAVE_T_MS[0]) & (t_ms <= SAVE_T_MS[1])]
print(f"saving null weights for {save_centers.size} windows ({SAVE_T_MS[0]}-{SAVE_T_MS[1]} ms)")

# =========================
# 4) Permutation test
# =========================
perm_result = mc.sliding_window_permutation_test(X_pix_frames_trials=X_roi, y_trials=y_trials,
                                                 make_estimator=make_estimator,
                                                 window_size=window_size, start_frame=start_frame,
                                                 stop_frame=stop_frame, step=step, n_splits=n_splits,
                                                 n_perm=N_PERMUTATIONS, seed=PERM_SEED,
                                                 return_null=True, return_perm_stats=False,
                                                 save_weight_centers=save_centers)

# =========================
# 5) Save
# =========================
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
token = RUN_DIR.name.split("__")[1]          # e.g. 110209a24_frame1-100
save_dir = PERM_ROOT / f"slidingwindow__perm{N_PERMUTATIONS}__{token}__withWeights__{ts}"
save_dir.mkdir(parents=True, exist_ok=False)

perm_config = {"source_run_dir": str(RUN_DIR),
               "source_model_name": RUN_DIR.name,
               "n_permutations": int(N_PERMUTATIONS),
               "perm_seed": int(PERM_SEED),
               "baseline_frames": list(baseline),
               "window_size": window_size, "start_frame": start_frame, "stop_frame": stop_frame,
               "step": step, "n_splits": n_splits,
               "svm_random_state": SVM_SEED,
               "save_t_ms": list(SAVE_T_MS)}
with open(save_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(perm_config, f, indent=2)

np.savez_compressed(save_dir / "perm_result.npz",
                    centers=np.asarray(perm_result["centers"], dtype=int),
                    real_trial_curve=np.asarray(real_results["trial_acc_mean"], dtype=float),
                    real_fold_trial_acc=np.asarray(real_results["fold_trial_acc"], dtype=float),   # (n_windows, n_folds)
                    real_frame_curve=np.asarray(real_results["frame_acc_mean"], dtype=float),
                    null_trial_acc=np.asarray(perm_result["null_trial_acc"], dtype=float),         # (n_perm, n_windows)
                    null_frame_acc=np.asarray(perm_result["null_frame_acc"], dtype=float),         # (n_perm, n_windows)
                    null_trial_folds=np.asarray(perm_result["null_trial_folds"], dtype=float),     # (n_perm, n_windows, n_folds)
                    null_trial_mean=np.asarray(perm_result["null_trial_acc_mean"], dtype=float),
                    null_trial_std=np.asarray(perm_result["null_trial_acc_std"], dtype=float),
                    null_frame_mean=np.asarray(perm_result["null_frame_acc_mean"], dtype=float),
                    null_frame_std=np.asarray(perm_result["null_frame_acc_std"], dtype=float),
                    null_w_mean=perm_result["null_w_mean"],                                        # (n_perm, n_save, n_roi_pixels)
                    null_w_centers=np.asarray(perm_result["null_w_centers"], dtype=int),           # (n_save,)
                    roi_mask=ROI_mask)                                                             # (10000,)

print("Saved to:", save_dir)
