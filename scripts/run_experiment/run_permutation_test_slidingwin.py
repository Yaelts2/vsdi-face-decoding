# run_permutation_test_slidingwin.py
# Sliding-window permutation test for a LIST of models, ONLY on the windows from stimulus onset to +300 ms
# (both phase peaks are inside), saving the weights of every shuffled model
# (for the phase-1 / phase-2 overlap null in MATLAB).
#
# Made to run alone on a separate computer:
#   - runs the models in RUN_NAMES one after the other
#   - if one model fails, the error is logged and the next model starts
#   - models that already have a finished "withWeights" run are skipped (safe to restart after a crash)
#   - everything printed is also written to a log file in PERM_ROOT
#
# Run it from the project folder (data_dir in the model configs is relative to it):
#   cd C:\project\vsdi-face-decoding
#   .venv\Scripts\python.exe scripts\run_experiment\run_permutation_test_slidingwin.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import gc
import json
import os
import time
import traceback
import numpy as np
from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import model_control as mc
from scripts.functions_scripts import save_results as sr

# =========================
# CONFIG
# =========================
RESULTS_ROOT = Path(r"C:\project\vsdi-face-decoding\results")

# the model folders (inside RESULTS_ROOT) to run, in order   <- EDIT
RUN_NAMES = [
    "sliding_window__110209a24_frame1-100__SVM_10foldCV____2026-06-14_12-01-00",
    # "sliding_window__110209a15_frame1-100__SVM_10foldCV____....",
    # ...
]

SKIP_DONE = True        # skip a model that already has a finished withWeights run

N_PERMUTATIONS = 100
PERM_SEED = 42
SVM_SEED = 42

# shuffle only the windows whose CENTER is in this range (ms from stimulus onset)
T_MS = (0, 300)
ZERO_FRAME = 27      # frame 27 (1-based) = python frame 26 = stimulus onset = 0 ms
FRAME_MS = 10

PERM_ROOT = RESULTS_ROOT / "permutation_test_sliding_window"


# =========================
# Log: everything printed goes to the screen AND to a log file
# =========================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


PERM_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = PERM_ROOT / f"batch_log__{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
_log_file = open(LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

T0 = time.perf_counter()


def stamp(msg):
    """print a step with the clock time and the time since the batch started"""
    el = time.perf_counter() - T0
    print(f"[{datetime.now():%H:%M:%S} | +{int(el // 3600)}h {int(el % 3600 // 60):02d}m] {msg}", flush=True)


def model_token(run_name):
    return run_name.split("__")[1]          # e.g. 110209a24_frame1-100


def find_done_run(run_name):
    """a finished withWeights run of this model (has perm_result.npz), or None"""
    for d in sorted(PERM_ROOT.glob(f"slidingwindow__perm{N_PERMUTATIONS}__{model_token(run_name)}__withWeights__*")):
        if (d / "perm_result.npz").exists():
            return d
    return None


# =========================
# One model
# =========================
def run_one(run_dir):
    # 1) Load the real model
    stamp(f"loading model {run_dir.name}")
    config, real_results, ROI_mask_path = sr.load_experiment(run_dir)

    face_file = config["face_file"]
    nonface_file = config["nonface_file"]
    data_dir = config["data_dir"]
    baseline = tuple(config["zscore_baseline_frames"])

    window_size = int(config.get("window_size", 5))
    step = int(config.get("step", 1))
    n_splits = int(config.get("n_splits", 10))

    make_estimator = lambda: cv.make_linear_svm(C=0.0001, max_iter=100000, random_state=SVM_SEED)

    # 2) Rebuild the data exactly as in training
    stamp("loading + z-scoring data")
    X_trials, y_trials, _ = pre.build_X_y(face_file=face_file, nonface_file=nonface_file, data_dir=data_dir)
    X_z, _mean, _std = pre.zscore_dataset_pixelwise_trials(X_trials, baseline_frames=baseline)
    ROI_mask = np.asarray(np.load(ROI_mask_path), dtype=bool).ravel()
    X_roi = X_z[ROI_mask, :, :]
    del X_trials, X_z
    stamp(f"data ready: X_roi {X_roi.shape}, {y_trials.size} trials")

    # 3) Windows to shuffle: centers from stimulus onset to +300 ms
    centers_all = np.asarray(real_results["centers"], dtype=int)          # python frame of each window center
    t_ms_all = (centers_all + 1 - ZERO_FRAME) * FRAME_MS
    keep = (t_ms_all >= T_MS[0]) & (t_ms_all <= T_MS[1])
    perm_centers = centers_all[keep]                                        # 26 ... 56
    perm_start = int(perm_centers[0]) - window_size // 2                    # first window start (24)
    perm_stop = int(perm_centers[-1]) - window_size // 2 + window_size      # end of last window (59)
    stamp(f"shuffling {perm_centers.size} windows: centers {perm_centers[0]}-{perm_centers[-1]} "
          f"({T_MS[0]}-{T_MS[1]} ms), frames {perm_start}-{perm_stop}")

    # 4) Permutation test
    perm_result = mc.sliding_window_permutation_test(X_pix_frames_trials=X_roi, y_trials=y_trials,
                                                     make_estimator=make_estimator,
                                                     window_size=window_size, start_frame=perm_start,
                                                     stop_frame=perm_stop, step=step, n_splits=n_splits,
                                                     n_perm=N_PERMUTATIONS, seed=PERM_SEED,
                                                     return_null=True, return_perm_stats=False,
                                                     save_weight_centers=perm_centers)
    if not np.array_equal(perm_result["centers"], perm_centers):
        raise RuntimeError("window centers of the permutation run differ from the real model - check start/stop")
    stamp("permutation test done")

    # 5) Save: write into a "__running" folder, rename when complete
    #    (so a crash mid-save never looks like a finished run)
    stamp("saving")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_dir = PERM_ROOT / f"slidingwindow__perm{N_PERMUTATIONS}__{model_token(run_dir.name)}__withWeights__{ts}"
    tmp_dir = final_dir.with_name(final_dir.name + "__running")
    tmp_dir.mkdir(parents=True, exist_ok=False)

    perm_config = {"source_run_dir": str(run_dir),
                   "source_model_name": run_dir.name,
                   "n_permutations": int(N_PERMUTATIONS),
                   "perm_seed": int(PERM_SEED),
                   "baseline_frames": list(baseline),
                   "window_size": window_size, "step": step, "n_splits": n_splits,
                   "perm_start_frame": perm_start, "perm_stop_frame": perm_stop,
                   "t_ms": list(T_MS), "zero_frame": ZERO_FRAME, "frame_ms": FRAME_MS,
                   "svm_random_state": SVM_SEED}
    with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(perm_config, f, indent=2)

    np.savez_compressed(tmp_dir / "perm_result.npz",
                        # everything below is ONLY for the shuffled windows (0-300 ms)
                        centers=np.asarray(perm_result["centers"], dtype=int),                         # (n_windows,)
                        real_trial_curve=np.asarray(real_results["trial_acc_mean"], dtype=float)[keep],
                        real_fold_trial_acc=np.asarray(real_results["fold_trial_acc"], dtype=float)[keep],   # (n_windows, n_folds)
                        real_frame_curve=np.asarray(real_results["frame_acc_mean"], dtype=float)[keep],
                        null_trial_acc=np.asarray(perm_result["null_trial_acc"], dtype=float),         # (n_perm, n_windows)
                        null_frame_acc=np.asarray(perm_result["null_frame_acc"], dtype=float),         # (n_perm, n_windows)
                        null_trial_folds=np.asarray(perm_result["null_trial_folds"], dtype=float),     # (n_perm, n_windows, n_folds)
                        null_trial_mean=np.asarray(perm_result["null_trial_acc_mean"], dtype=float),
                        null_trial_std=np.asarray(perm_result["null_trial_acc_std"], dtype=float),
                        null_frame_mean=np.asarray(perm_result["null_frame_acc_mean"], dtype=float),
                        null_frame_std=np.asarray(perm_result["null_frame_acc_std"], dtype=float),
                        null_w_mean=perm_result["null_w_mean"],                                        # (n_perm, n_windows, n_roi_pixels)
                        null_w_centers=np.asarray(perm_result["null_w_centers"], dtype=int),           # (n_windows,) = centers
                        roi_mask=ROI_mask)                                                             # (10000,)
    os.replace(tmp_dir, final_dir)
    stamp(f"saved to {final_dir}")
    return final_dir


# =========================
# Batch
# =========================
if __name__ == "__main__":
    stamp(f"batch of {len(RUN_NAMES)} models, log: {LOG_PATH}")
    status = []          # (name, 'done' / 'skipped' / 'FAILED', info)

    # check all paths first, so a typo is found now and not hours later
    for name in RUN_NAMES:
        if not (RESULTS_ROOT / name).exists():
            print(f"  WARNING: model folder not found, it will fail: {RESULTS_ROOT / name}")

    for i, name in enumerate(RUN_NAMES, start=1):
        print("\n" + "=" * 90)
        stamp(f"MODEL {i}/{len(RUN_NAMES)}: {name}")
        print("=" * 90)

        done = find_done_run(name) if SKIP_DONE else None
        if done is not None:
            stamp(f"skipped - already done: {done.name}")
            status.append((name, "skipped", done.name))
            continue

        t_model = time.perf_counter()
        try:
            out = run_one(RESULTS_ROOT / name)
            mins = (time.perf_counter() - t_model) / 60
            status.append((name, "done", f"{mins:.0f} min -> {out.name}"))
        except Exception as e:
            print(f"\n  !!! FAILED: {name}\n  {type(e).__name__}: {e}")
            traceback.print_exc()
            status.append((name, "FAILED", f"{type(e).__name__}: {e}"))
        gc.collect()

    print("\n" + "=" * 90)
    stamp("BATCH FINISHED")
    for name, st, info in status:
        print(f"  {st:<8} {model_token(name):<25} {info}")
    print(f"\nlog saved to {LOG_PATH}")
