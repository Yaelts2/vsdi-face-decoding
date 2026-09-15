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
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import ml_plots as pl
from scripts.functions_scripts import sliding_win as sw
from scripts.functions_scripts import feature_extraction as fe

ourCmap = pre.green_gray_magenta()


# =========================
# CONFIG -- PILOT: reduced overlap (step=3), session 030209_e_1,5, V1+V2
# =========================
face_file = "condsXn1_110209b.npy"
nonface_file = "condsXn5_110209b.npy"
data_dir = "data/processed/condsXn/"

Baseline_frames_zscore = (1, 24)

# TODO: confirm these are the correct V1/V2-specific ROI masks for date 030209
ROI_MASK_PATH_V1 = "data/processed/v1_mask_110209.npy"
ROI_MASK_PATH_V2 = "data/processed/v2_mask_110209.npy"

SEED = 42
model = lambda: cv.make_linear_svm(C=0.0001, max_iter=100000, random_state=SEED)
window_size = int(5)
start_frame = int(1)
stop_frame = int(100)
step = int(3)   # PILOT CHANGE: was 1 (80% overlap) -> now 3 (40% overlap)
n_splits = int(10)

ZERO_FRAME = 27
FRAME_DURATION_MS = 10  # per raw frame (unchanged -- this is a property of the recording, not the window)

PHASE1_T_START, PHASE1_T_STOP = 0, 130
PHASE2_T_START, PHASE2_T_STOP = 130, 250

LAG_MIN = -90
LAG_MAX = 90
# LAG_STEP will be set dynamically to match actual window spacing (see below)


# ================== prepare data (ONCE, shared by V1 and V2) ==================
X_trials, y_trials, dataset_info = pre.build_X_y(face_file=face_file,
                                                  nonface_file=nonface_file,
                                                  data_dir=data_dir)
print(f"Dataset shape: {X_trials.shape}, Labels shape: {y_trials.shape}")

X_z, mean, std = pre.zscore_dataset_pixelwise_trials(X_trials, Baseline_frames_zscore)
print("Data z-scored across all trials (shared by V1 and V2).")


def centers_to_time_ms(centers):
    return (np.asarray(centers) - ZERO_FRAME) * FRAME_DURATION_MS


def run_decode(X_z, roi_mask_path, area_name):
    roi_mask = np.load(roi_mask_path).astype(bool)
    print(f"{area_name} ROI mask loaded: {roi_mask.shape}, {roi_mask.sum()} pixels")
    X_roi = X_z[roi_mask, :, :]
    print(f"{area_name} ROI selected: {X_roi.shape}")

    results = sw.sliding_window_decode_with_stats(X_roi, y_trials, model,
                                                    window_size, start_frame,
                                                    stop_frame, step, n_splits)
    results["zscore_std_pooled"] = std
    print(f"{area_name}: peak trial accuracy = {results['trial_acc_mean'].max():.3f}")
    return results


results_v1 = run_decode(X_z, ROI_MASK_PATH_V1, "V1")
results_v2 = run_decode(X_z, ROI_MASK_PATH_V2, "V2")

# -----------------------------------------------------------------
# Figure: frame-level vs trial-level accuracy across time, V1 and V2
# -----------------------------------------------------------------
# NOTE: the underlying SVM classification happens at the FRAME level
# (every frame within a window is its own train/test sample -- see
# fe.frames_as_samples in the manual-score-check block above).
# frame_acc_mean/std = raw per-frame accuracy.
# trial_acc_mean/std = accuracy after aggregating each trial's frames
#   within a window into one trial-level call -- this is the level
#   oof_trial_score (and therefore Method B) actually operates on.
v1_time_ms_acc = centers_to_time_ms(results_v1["centers"])
v2_time_ms_acc = centers_to_time_ms(results_v2["centers"])

figAcc, axesAcc = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, results, time_ms_acc, area_name in zip(
        axesAcc, [results_v1, results_v2], [v1_time_ms_acc, v2_time_ms_acc], ["V1", "V2"]):
    ax.plot(time_ms_acc, results["frame_acc_mean"], label="frame-level accuracy",
             color="tab:cyan", linewidth=1.5)
    ax.fill_between(time_ms_acc,
                      results["frame_acc_mean"] - results["frame_acc_std"],
                      results["frame_acc_mean"] + results["frame_acc_std"],
                      color="tab:cyan", alpha=0.15)
    ax.plot(time_ms_acc, results["trial_acc_mean"], label="trial-level accuracy",
             color="tab:red", linewidth=1.5)
    ax.fill_between(time_ms_acc,
                      results["trial_acc_mean"] - results["trial_acc_std"],
                      results["trial_acc_mean"] + results["trial_acc_std"],
                      color="tab:red", alpha=0.15)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.axvspan(PHASE1_T_START, PHASE1_T_STOP, color="gray", alpha=0.1)
    ax.axvspan(PHASE2_T_START, PHASE2_T_STOP, color="purple", alpha=0.08)
    ax.set_xlabel("time (ms)")
    ax.set_title(f"{area_name} accuracy across time (step={step})")
    ax.legend(fontsize=8)
axesAcc[0].set_ylabel("accuracy")
figAcc.suptitle("Frame-level vs. trial-level decoding accuracy — 030209_e_1,5 pilot")
figAcc.tight_layout()


# ================== Method B, in-memory, no save/reload ==================
def get_phase_indices(time_ms, t_start, t_stop):
    mask = (time_ms >= t_start) & (time_ms <= t_stop)
    return np.where(mask)[0]


def compute_lag_curve(v1_scores, v2_scores, n_windows, ref_idx,
                       lag_min, lag_max, lag_step, index_step_ms):
    """index_step_ms: actual ms spacing between consecutive window centers
    (computed from the real centers array, not assumed)."""
    n_trials = v2_scores.shape[1]
    assert v1_scores.shape[1] == n_trials, "V1/V2 trial count mismatch!"

    lags = np.arange(lag_min, lag_max + lag_step, lag_step)
    mean_r_list, surviving_lags = [], []

    for lag in lags:
        shift = lag / index_step_ms
        if not np.isclose(shift, round(shift)):
            continue  # lag isn't a clean multiple of the actual window spacing -- skip
        shift = int(round(shift))
        v1_idx = ref_idx + shift
        if v1_idx.min() < 0 or v1_idx.max() > n_windows - 1:
            continue
        v2_vals = v2_scores[ref_idx, :]
        v1_vals = v1_scores[v1_idx, :]
        trial_rs = np.full(n_trials, np.nan)
        for tr in range(n_trials):
            v2_t, v1_t = v2_vals[:, tr], v1_vals[:, tr]
            if np.std(v2_t) == 0 or np.std(v1_t) == 0:
                continue
            r, _ = pearsonr(v2_t, v1_t)
            trial_rs[tr] = r
        mean_r_list.append(np.nanmean(trial_rs))
        surviving_lags.append(lag)

    return np.array(surviving_lags), np.array(mean_r_list)


v1_scores = results_v1["oof_trial_score"]
v2_scores = results_v2["oof_trial_score"]
print(f"\nV1 scores shape: {v1_scores.shape}, V2 scores shape: {v2_scores.shape}")

v1_time_ms = centers_to_time_ms(results_v1["centers"])
v2_time_ms = centers_to_time_ms(results_v2["centers"])
if not np.array_equal(v1_time_ms, v2_time_ms):
    raise ValueError("V1 and V2 centers differ -- stopping (shared-window assumption broken)")
time_ms = v1_time_ms
n_windows = len(time_ms)

# actual spacing between consecutive windows, in ms -- should be step * FRAME_DURATION_MS = 30
index_step_ms = float(np.median(np.diff(time_ms)))
print(f"Window spacing: {index_step_ms:.1f} ms (expected: {step * FRAME_DURATION_MS} ms)")
LAG_STEP = index_step_ms  # lag grid must move in units of the actual window spacing

phases = [
    ("phase 1", PHASE1_T_START, PHASE1_T_STOP),
    ("phase 2", PHASE2_T_START, PHASE2_T_STOP),
]
phase_results = {}
for phase_name, t_start, t_stop in phases:
    ref_idx = get_phase_indices(time_ms, t_start, t_stop)
    ref_times = time_ms[ref_idx]
    print(f"\n{phase_name} ({t_start}-{t_stop}ms) reference timepoints (ms): "
          f"{ref_times}  ({len(ref_idx)} points)")

    lags, mean_r = compute_lag_curve(v1_scores, v2_scores, n_windows, ref_idx,
                                       LAG_MIN, LAG_MAX, LAG_STEP, index_step_ms)
    print(f"{'lag (ms)':>10} {'mean r':>10}")
    for lag, r in zip(lags, mean_r):
        marker = "  <-- max" if r == np.nanmax(mean_r) else ""
        print(f"{lag:>10.0f} {r:>10.4f}{marker}")

    best_lag = lags[np.nanargmax(mean_r)]
    print(f"{phase_name}: best lag = {best_lag:+.0f} ms "
          f"({'V2 leads V1' if best_lag > 0 else 'V1 leads V2' if best_lag < 0 else 'no lead'})")

    phase_results[phase_name] = {"lags": lags, "mean_r": mean_r, "best_lag": best_lag}


# -----------------------------------------------------------------
# Figure: phase 1 and phase 2 lag curves, side by side
# -----------------------------------------------------------------
figC, axesC = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (phase_name, t_start, t_stop) in zip(axesC, phases):
    res = phase_results[phase_name]
    ax.plot(res["lags"], res["mean_r"], marker="o")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(res["best_lag"], color="red", linestyle=":", linewidth=1.5,
               label=f"best lag = {res['best_lag']:+.0f} ms")
    ax.set_xlabel("lag (ms)  [positive = V2 leads V1]")
    ax.set_title(f"{phase_name} ({t_start}-{t_stop}ms)")
    ax.legend()
axesC[0].set_ylabel("mean per-trial correlation (r)")
figC.suptitle(f"Method B PILOT (step={step}, {index_step_ms:.0f}ms spacing) — 030209_e_1,5")
figC.tight_layout()

plt.show()