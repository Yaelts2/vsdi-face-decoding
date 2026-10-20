# run_permutation_test_slidingwin.py
"""
Sliding-window label-permutation test for ONE single-session model, now also keeping
the WEIGHTS of every shuffled model, so the phase-1 / phase-2 extreme-weight overlap
can be recomputed on each shuffled model in MATLAB (shuffle-model null for the overlap).

What is saved per permutation (in addition to the accuracy null, unchanged):
    null_w_mean : (n_perm, n_save_windows, n_roi_pixels) float32
                  = w_mean_windows of the shuffled model (mean over the 10 folds),
                    i.e. exactly the kind of weights that Weights_acrossT holds for the real model.
    Only windows with centers inside SAVE_T_MS are kept (all 95 windows would be ~300 MB / session).

Run once per dataset: edit RUN_DIR (and OLD_PERM_DIR if you have an old run for it), run,
then open the result with open_permutation_sliding.py, which appends everything to the model .mat.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
print("Project root:", PROJECT_ROOT)

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

# the REAL sliding-window model of this dataset
RUN_DIR = RESULTS_ROOT / "sliding_window__110209a24_frame1-100__SVM_10foldCV____2026-06-14_12-01-00"   # <- EDIT

# the OLD permutation folder of the same model (no weights), for check (c): with the same seed the
# label permutations are identical, so the new null accuracies must equal the old ones exactly.
#   "auto" = search PERM_ROOT for an old run whose config.json points to RUN_DIR (recommended)
#   None   = skip the check (e.g. 03/02 datasets that have no old run)
#   or a full path to the folder
OLD_PERM_DIR = "auto"

N_PERMUTATIONS = 100
PERM_SEED = 42          # keep 42 = same shuffles as the existing runs
SVM_SEED = 42

# time range (ms from stimulus onset) of the windows whose null weights are kept.
# Must contain every window MATLAB uses for phase 1 and phase 2 (incl. the +-1 windows around the peaks).
SAVE_T_MS = (0, 300)
ZERO_FRAME = 27         # frame 27 = 0 ms (same convention as the .mat export)
FRAME_MS = 10

PERM_ROOT = RESULTS_ROOT / "permutation_test_sliding_window"

# =========================
# 0) Resolve OLD_PERM_DIR NOW (fail fast - before hours of training, not after)
# =========================
if not RUN_DIR.exists():
    raise FileNotFoundError(f"RUN_DIR not found: {RUN_DIR}")

if isinstance(OLD_PERM_DIR, str) and OLD_PERM_DIR == "auto":
    matches = []
    for cfg_path in sorted(PERM_ROOT.glob("*/config.json")):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                c = json.load(f)
        except Exception:
            continue
        same_model = Path(str(c.get("source_run_dir", ""))).name == RUN_DIR.name
        if same_model and "null_weights" not in c and (cfg_path.parent / "perm_result.npz").exists():
            matches.append((int(c.get("n_permutations", 0)), int(c.get("perm_seed", -1)), cfg_path.parent))
    # must be the same seed; prefer the run with the same number of permutations
    matches = [m for m in matches if m[1] == PERM_SEED]
    same_n = [m for m in matches if m[0] == N_PERMUTATIONS]
    OLD_PERM_DIR = (same_n or matches)[-1][2] if matches else None
    print("Old permutation run for check (c):", OLD_PERM_DIR.name if OLD_PERM_DIR else "none found -> check skipped")
elif OLD_PERM_DIR is not None:
    OLD_PERM_DIR = Path(OLD_PERM_DIR)
    if not (OLD_PERM_DIR / "perm_result.npz").exists():
        raise FileNotFoundError(f"OLD_PERM_DIR has no perm_result.npz: {OLD_PERM_DIR}")

# =========================
# 1) Load the real model (config + results + ROI path)
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
print(f"  window_size={window_size} start={start_frame} stop={stop_frame} step={step} n_splits={n_splits}")

make_estimator = lambda: cv.make_linear_svm(C=0.0001, max_iter=100000, random_state=SVM_SEED)

# =========================
# 2) Rebuild the data exactly as in training
# =========================
X_trials, y_trials, _ = pre.build_X_y(face_file=face_file, nonface_file=nonface_file, data_dir=data_dir)
X_z, _mean, _std = pre.zscore_dataset_pixelwise_trials(X_trials, baseline_frames=baseline)
ROI_mask = np.asarray(np.load(ROI_mask_path), dtype=bool).ravel()
X_roi = X_z[ROI_mask, :, :]
print("  X_roi:", X_roi.shape, " y:", y_trials.shape, " ROI pixels:", int(ROI_mask.sum()))

# =========================
# 3) Which windows keep their null weights
# =========================
centers_real = np.asarray(real_results["centers"], dtype=int)
t_ms_real = (centers_real + 1 - ZERO_FRAME) * FRAME_MS      # same time axis as the plots (frame = center+1)
save_centers = centers_real[(t_ms_real >= SAVE_T_MS[0]) & (t_ms_real <= SAVE_T_MS[1])]
print(f"  keeping null weights for {save_centers.size} windows "
      f"({(save_centers[0] + 1 - ZERO_FRAME) * FRAME_MS} to {(save_centers[-1] + 1 - ZERO_FRAME) * FRAME_MS} ms)")
est_mb = N_PERMUTATIONS * save_centers.size * X_roi.shape[0] * 4 / 1e6
print(f"  null weights size ~{est_mb:.0f} MB (float32, before compression)")

# =========================
# 4) Permutation test (same pipeline as the real model) + null weights
# =========================
perm_result = mc.sliding_window_permutation_test(X_pix_frames_trials=X_roi, y_trials=y_trials,
                                                 make_estimator=make_estimator,
                                                 window_size=window_size, start_frame=start_frame,
                                                 stop_frame=stop_frame, step=step, n_splits=n_splits,
                                                 n_perm=N_PERMUTATIONS, seed=PERM_SEED,
                                                 return_null=True, return_perm_stats=False,
                                                 save_weight_centers=save_centers)

# =========================
# 5) Sanity checks
# =========================
# (a) the re-fitted real model must give the same weights as the saved real model
save_idx_real = np.array([int(np.where(centers_real == c)[0][0]) for c in save_centers])
w_saved = np.asarray(real_results["w_mean_windows"], dtype=np.float32)[save_idx_real]
d_real = float(np.max(np.abs(perm_result["real_w_mean"] - w_saved)))
print(f"\nCheck (a) real weights re-fit vs saved model: max abs diff = {d_real:.3e} "
      f"(weights scale ~{float(np.max(np.abs(w_saved))):.3e})")
if d_real > 1e-3 * float(np.max(np.abs(w_saved))):
    print("  WARNING: the re-fitted real model differs from the saved one -> data/preprocessing/ROI "
          "is not identical to training. Do NOT trust the null until this is fixed.")

# (b) the null weights really are shuffled-model weights: peak null accuracy should be ~chance
print(f"Check (b) null trial acc: mean {float(perm_result['null_trial_acc'].mean()):.3f}, "
      f"max of window means {float(perm_result['null_trial_acc'].mean(axis=0).max()):.3f} (should be ~0.5)")

# (c) same shuffles as the old run (never allowed to crash the run before saving)
d_old = None
if OLD_PERM_DIR is not None:
    try:
        old = np.load(Path(OLD_PERM_DIR) / "perm_result.npz", allow_pickle=True)
        a_old, a_new = np.asarray(old["null_trial_acc"]), np.asarray(perm_result["null_trial_acc"])
        n = min(a_old.shape[0], a_new.shape[0])          # old run may have more permutations (e.g. 1000)
        d_old = float(np.max(np.abs(a_old[:n] - a_new[:n])))
        print(f"Check (c) null trial acc vs old run ({n} perms compared): max abs diff = {d_old:.3e} (should be 0)")
        if d_old > 1e-9:
            print("  WARNING: the shuffles differ from the old run (different seed/data/sklearn version?).")
    except Exception as e:
        print(f"Check (c) skipped - could not compare with the old run: {e}")

# =========================
# 6) Save (folder name carries the model token, so open_permutation_sliding.py finds the model .mat)
# =========================
PERM_ROOT.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = RUN_DIR.name
token = model_name.split("__")[1] if "__" in model_name else "model"     # e.g. 110209a24_frame1-100
save_dir = PERM_ROOT / f"slidingwindow__perm{N_PERMUTATIONS}__{token}__withWeights__{ts}"
save_dir.mkdir(parents=True, exist_ok=False)

perm_config = {"source_run_dir": str(RUN_DIR),
               "source_model_name": model_name,
               "roi_mask_path": str(ROI_mask_path),
               "n_permutations": int(N_PERMUTATIONS),
               "perm_seed": int(PERM_SEED),
               "baseline_frames": list(baseline),
               "window_size": window_size, "start_frame": start_frame, "stop_frame": stop_frame,
               "step": step, "n_splits": n_splits,
               "estimator": config.get("model", None),
               "svm_random_state": SVM_SEED,
               "null_weights": "w_mean_windows (mean over folds) of each shuffled model, ROI pixels only",
               "save_t_ms": list(SAVE_T_MS), "zero_frame": ZERO_FRAME, "frame_ms": FRAME_MS,
               "check_real_weights_maxdiff": d_real,
               "old_perm_dir": str(OLD_PERM_DIR) if OLD_PERM_DIR else None,
               "check_old_null_maxdiff": d_old}
with open(save_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(perm_config, f, indent=2)

np.savez_compressed(save_dir / "perm_result.npz",
                    centers=np.asarray(perm_result["centers"], dtype=int),
                    # real curves from the saved model
                    real_trial_curve=np.asarray(real_results["trial_acc_mean"], dtype=float),
                    real_fold_trial_acc=np.asarray(real_results["fold_trial_acc"], dtype=float),   # (n_windows, n_folds)
                    real_frame_curve=np.asarray(real_results["frame_acc_mean"], dtype=float),
                    # accuracy null (unchanged)
                    null_trial_acc=np.asarray(perm_result["null_trial_acc"], dtype=float),         # (n_perm, n_windows)
                    null_frame_acc=np.asarray(perm_result["null_frame_acc"], dtype=float),         # (n_perm, n_windows)
                    null_trial_folds=np.asarray(perm_result["null_trial_folds"], dtype=float),     # (n_perm, n_windows, n_folds)
                    null_trial_mean=np.asarray(perm_result["null_trial_acc_mean"], dtype=float),
                    null_trial_std=np.asarray(perm_result["null_trial_acc_std"], dtype=float),
                    null_frame_mean=np.asarray(perm_result["null_frame_acc_mean"], dtype=float),
                    null_frame_std=np.asarray(perm_result["null_frame_acc_std"], dtype=float),
                    # NEW: null weights
                    null_w_mean=perm_result["null_w_mean"],                                        # (n_perm, n_save, n_roi) float32
                    null_w_centers=np.asarray(perm_result["null_w_centers"], dtype=int),           # (n_save,)
                    roi_mask=ROI_mask)                                                             # (10000,) bool

print("\nSaved sliding-window permutation results (with null weights) to:")
print(" ", save_dir)
