"""
test_errorTrials_V2.py

Driver script for Aim 3, Part 2 -- V2 AREA MODELS. Identical structure to
test_errorTrials_allareas.py / test_errorTrials_V1.py, just pointed at the
V2-trained sliding-window decoders and a separate V2 cache, so V1/V2/allareas
results never collide.

Loops over sessions:
  1. Loads that session's already-trained sliding-window decoder (results)
  2. Processes that session's error trials against it
  3. Collects per-session results (accuracy, score)
  4. Builds grand averages across sessions (accuracy, score)
  5. Runs group-level significance tests (per-timepoint + windowed)
  6. Runs sliding-window Cohen's d (pooled, session-level) for both metrics

Results of the (slow) per-session loop are cached to disk. Set USE_CACHE=False
whenever you change anything UPSTREAM of the grand-average section (session
list, process_error_trials_for_session args, MATCH_CORRECT_TO_ERROR, etc).
Everything below the loop (stats, plotting, window choice) reruns in
seconds regardless of the cache.
"""

import matplotlib
matplotlib.use('TkAgg')

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scripts.functions_scripts import errorTrials_Functions as etf
from scripts.functions_scripts.save_results import load_experiment, load_data_from_config

print("Backend after all imports:", matplotlib.get_backend())

# =====================================================================
# CACHE + ANALYSIS CONFIG
# =====================================================================

CACHE_PATH = Path(r"C:\project\vsdi-face-decoding\results\error_trials_cache_V2.pkl")
USE_CACHE = False      # set False to force a full recompute of the per-session loop

# If True, correct trials are randomly subsampled ONCE per session (before any
# window is computed) down to that session's error-trial count -- matches
# Ayzenshtat et al. (2012)'s approach so both groups contribute equal noise.
# Changing this REQUIRES USE_CACHE=False for one run (it changes what the loop computes).
MATCH_CORRECT_TO_ERROR = True
SUBSAMPLE_SEED = 42

# =====================================================================
# CONFIG -- one entry per session (V2-trained decoders)
# =====================================================================

sessions = [
    {
        "tag": "110209a15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209a15_V2_frame1-100__SVM_10foldCV____2026-08-06_10-52-10",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209a\errorTrialsData_110209a15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209a24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209a24_V2_frame1-100__SVM_10foldCV____2026-08-06_10-57-43",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209a\errorTrialsData_110209a24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "110209b15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209b15_V2_frame1-100__SVM_10foldCV____2026-08-06_10-59-53",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209b\errorTrialsData_110209b15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209b24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209b24_V2_frame1-100__SVM_10foldCV____2026-08-06_11-01-53",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209b\errorTrialsData_110209b24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "110209c15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209c15_V2_frame1-100__SVM_10foldCV____2026-08-06_11-03-45",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209c\errorTrialsData_110209c15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209c24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209c24_V2_frame1-100__SVM_10foldCV____2026-08-06_11-05-39",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209c\errorTrialsData_110209c24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "110209d15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209d15_V2_frame1-100__SVM_10foldCV____2026-08-06_11-15-41",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209d\errorTrialsData_110209d15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209d24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__110209d24_V2_frame1-100__SVM_10foldCV____2026-08-06_11-18-22",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209d\errorTrialsData_110209d24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209a15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209a15_V2_frame1-100__SVM_10foldCV____2026-08-06_11-47-22",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209a\errorTrialsData_030209a15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209a24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209a24_V2_frame1-100__SVM_10foldCV____2026-08-06_11-48-32",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209a\errorTrialsData_030209a24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209c15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209c15_V2_frame1-100__SVM_10foldCV____2026-08-06_11-50-17",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209c\errorTrialsData_030209c15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209c24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209c24_V2_frame1-100__SVM_10foldCV____2026-08-06_11-51-26",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209c\errorTrialsData_030209c24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209e15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209e15_V2_frame1-100__SVM_10foldCV____2026-08-06_11-53-01",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209e\errorTrialsData_030209e15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209e24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209e24_V2_frame1-100__SVM_10foldCV____2026-08-06_11-54-20",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209e\errorTrialsData_030209e24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209f15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209f15_V2_frame1-100__SVM_10foldCV____2026-08-06_11-59-08",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209f\errorTrialsData_030209f15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209f24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\V2\sliding_window__030209f24_V2_frame1-100__SVM_10foldCV____2026-08-06_12-00-53",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209f\errorTrialsData_030209f24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
]

# =====================================================================
# PER-SESSION PIPELINE (cached -- see CACHE_PATH / USE_CACHE above)
# =====================================================================

if USE_CACHE and CACHE_PATH.exists():
    print(f"Loading cached results from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    error_acc_list = cache["error_acc_list"]
    correct_acc_list = cache["correct_acc_list"]
    error_score_list = cache["error_score_list"]
    correct_score_list = cache["correct_score_list"]
    session_names = cache["session_names"]
    centers_ref = cache["centers_ref"]
    per_session_results = cache["per_session_results"]

else:
    error_acc_list = []
    correct_acc_list = []
    error_score_list = []
    correct_score_list = []
    session_names = []
    centers_ref = None
    per_session_results = {}

    for s in sessions:
        print(f"\n===== Session: {s['tag']} =====")

        config, results, ROI_mask_path = load_experiment(s["run_dir"])
        X_z, y_trials = load_data_from_config(config)

        acc_results, score_results, meta, X_error_roi = etf.process_error_trials_for_session(
            error_mat_path=s["error_mat"],
            results=results,
            ROI_mask_path=ROI_mask_path,
            y_trials=y_trials,
            face_cond=s["face_cond"],
            nonface_cond=s["nonface_cond"],
            match_correct_to_error=MATCH_CORRECT_TO_ERROR,
            subsample_seed=SUBSAMPLE_SEED,
        )

        error_acc_list.append(acc_results["error_acc"])
        correct_acc_list.append(acc_results["correct_acc"])
        error_score_list.append(score_results["error_mean"])
        correct_score_list.append(score_results["correct_mean"])
        session_names.append(s["tag"])
        centers_ref = acc_results["centers"]

        per_session_results[s["tag"]] = {
            "results": results, "acc_results": acc_results, "score_results": score_results,
            "meta": meta, "X_error_roi": X_error_roi, "y_trials": y_trials,
        }

    print(f"Saving cache to {CACHE_PATH}")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "error_acc_list": error_acc_list, "correct_acc_list": correct_acc_list,
            "error_score_list": error_score_list, "correct_score_list": correct_score_list,
            "session_names": session_names, "centers_ref": centers_ref,
            "per_session_results": per_session_results,
        }, f)


# =====================================================================
# PER-SESSION FIGURES (accuracy + score, no stats -- just visual QC)
# =====================================================================

# for tag, sess_data in per_session_results.items():
#     etf.plot_accuracy_comparison(sess_data["acc_results"], session_tag=tag)
#     etf.plot_fold_averaged_comparison(sess_data["score_results"], session_tag=tag)

# =====================================================================
# GRAND AVERAGE + STATS -- ACCURACY
# =====================================================================

if len(error_acc_list) > 0:
    grand_results = etf.grand_average_correct_and_error(correct_acc_list, error_acc_list, session_names, centers_ref)
    fig, ax = etf.plot_grand_average_correct_and_error(grand_results)

    sig_results = etf.run_group_significance_tests(correct_acc_list, error_acc_list, centers_ref, null_value=0.5)

    for comp in ('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'):
        n_sig = sig_results[comp]['sig_mask'].sum()
        n_valid = (~np.isnan(sig_results[comp]['p_raw'])).sum()
        print(f"[ACC] {comp}: {n_sig} significant out of {n_valid} valid timepoints")

    etf.add_significance_markers(ax, sig_results, comparisons=('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'))

    # ---- windowed test: gap before phase 2 (session-mean based, with built-in plot) ----
    window_result_acc = etf.test_window_average(
        correct_acc_list, error_acc_list, centers_ref, window=(38, 45), null_value=0.5,
        label="accuracy", session_names=session_names, area_label="V2", ylabel="Accuracy")

    # =====================================================================
    # GRAND AVERAGE + STATS -- SCORE
    # =====================================================================
    grand_score_results = etf.grand_average_metric(
        correct_score_list, error_score_list, session_names, centers_ref, metric_name="score")
    fig2, ax2 = etf.plot_grand_average_metric(
        grand_score_results, ylabel="Mean score (+ = matches own true label)",
        title=f"Grand average across {grand_score_results['n_sessions']} sessions: correct vs. error label-consistent score (V2)",
        ref_line=0, ref_label="Decision boundary (0)")

    sig_score_results = etf.run_group_significance_tests(
        correct_score_list, error_score_list, centers_ref, null_value=0.0)

    for comp in ('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'):
        n_sig = sig_score_results[comp]['sig_mask'].sum()
        n_valid = (~np.isnan(sig_score_results[comp]['p_raw'])).sum()
        print(f"[SCORE] {comp}: {n_sig} significant out of {n_valid} valid timepoints")

    etf.add_significance_markers(ax2, sig_score_results,
        comparisons=('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'))

    # ---- windowed test: gap before phase 2 (session-mean based, with built-in plot) ----
    window_result_score = etf.test_window_average(
        correct_score_list, error_score_list, centers_ref, window=(38, 45), null_value=0.0,
        label="score", session_names=session_names, area_label="V2", ylabel="Score")

    # =====================================================================
    # SLIDING-WINDOW COHEN'S D (pooled, session-level) -- ACCURACY & SCORE
    # =====================================================================
    sliding_d_acc = etf.sliding_window_effect_size(correct_acc_list, error_acc_list, centers_ref,
                                                     window_width=16, method="pooled")
    etf.plot_sliding_window_effect_size(sliding_d_acc, area_label="Accuracy (V2)")

    sliding_d_score = etf.sliding_window_effect_size(correct_score_list, error_score_list, centers_ref,
                                                       window_width=16, method="pooled")
    etf.plot_sliding_window_effect_size(sliding_d_score, area_label="Score (V2)")

else:
    print("No sessions processed -- nothing to grand-average.")

# keep all figures open until you're done looking
plt.show(block=True)