"""
test_errorTrials_allareas.py

Driver script for Aim 3, Part 2 -- ALL-CHAMBER MODELS (formerly test_errorTrials.py).
Separated from V1/V2-specific versions so all three can be run independently
without cache collisions. See test_errorTrials_V1.py / test_errorTrials_V2.py
for the area-specific versions (identical structure, different run_dir/cache).

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

CACHE_PATH = Path(r"C:\project\vsdi-face-decoding\results\error_trials_cache_allareas.pkl")
USE_CACHE = True      # set False to force a full recompute of the per-session loop

# If True, correct trials are randomly subsampled ONCE per session (before any
# window is computed) down to that session's error-trial count -- matches
# Ayzenshtat et al. (2012)'s approach so both groups contribute equal noise.
# Changing this REQUIRES USE_CACHE=False for one run (it changes what the loop computes).
MATCH_CORRECT_TO_ERROR = False
SUBSAMPLE_SEED = 42

# =====================================================================
# CONFIG -- one entry per session (all-chamber models)
# =====================================================================

sessions = [
    {
        "tag": "110209a15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209a15_frame1-100__SVM_10foldCV____2026-07-19_12-00-22",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209a\errorTrialsData_110209a15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209a24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209a24_frame1-100__SVM_10foldCV____2026-07-19_13-24-35",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209a\errorTrialsData_110209a24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "110209b15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209b15_frame1-100__SVM_10foldCV____2026-07-19_13-44-39",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209b\errorTrialsData_110209b15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209b24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209b24_frame1-100__SVM_10foldCV____2026-07-20_11-30-27",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209b\errorTrialsData_110209b24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "110209c15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209c15_frame1-100__SVM_10foldCV____2026-07-20_11-42-35",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209c\errorTrialsData_110209c15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209c24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209c24_frame1-100__SVM_10foldCV____2026-07-20_11-54-00",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209c\errorTrialsData_110209c24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "110209d15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209d15_frame1-100__SVM_10foldCV____2026-07-20_15-33-10",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209d\errorTrialsData_110209d15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "110209d24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__110209d24_frame1-100__SVM_10foldCV____2026-07-26_10-19-07",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\110209d\errorTrialsData_110209d24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209a15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209a15_frame1-100__SVM_10foldCV____2026-07-26_12-08-59",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209a\errorTrialsData_030209a15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209a24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209a24_frame1-100__SVM_10foldCV____2026-07-26_12-16-43",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209a\errorTrialsData_030209a24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209c15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209c15_frame1-100__SVM_10foldCV____2026-07-26_12-42-22",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209c\errorTrialsData_030209c15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209c24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209c24_frame1-100__SVM_10foldCV____2026-07-26_12-34-26",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209c\errorTrialsData_030209c24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209e15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209e15_frame1-100__SVM_10foldCV____2026-07-26_12-51-26",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209e\errorTrialsData_030209e15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209e24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209e24_frame1-100__SVM_10foldCV____2026-07-26_13-49-07",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209e\errorTrialsData_030209e24.mat",
        "face_cond": 2,
        "nonface_cond": 4,
    },
    {
        "tag": "030209f15",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209f15_frame1-100__SVM_10foldCV____2026-07-27_10-41-55",
        "error_mat": r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials\030209f\errorTrialsData_030209f15.mat",
        "face_cond": 1,
        "nonface_cond": 5,
    },
    {
        "tag": "030209f24",
        "run_dir": r"C:\project\vsdi-face-decoding\results\allareas\sliding_window__030209f24_frame1-100__SVM_10foldCV____2026-07-27_11-01-09",
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
        correct_acc_list, error_acc_list, centers_ref, window=(41, 45), null_value=0.5,
        label="accuracy", session_names=session_names, area_label=None, ylabel="Accuracy")

    # =====================================================================
    # GRAND AVERAGE + STATS -- SCORE
    # =====================================================================
    grand_score_results = etf.grand_average_metric(
        correct_score_list, error_score_list, session_names, centers_ref, metric_name="score")
    fig2, ax2 = etf.plot_grand_average_metric(
        grand_score_results, ylabel="Mean score (+ = matches own true label)",
        title=f"Grand average across {grand_score_results['n_sessions']} sessions: correct vs. error label-consistent score",
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
        correct_score_list, error_score_list, centers_ref, window=(41, 45), null_value=0.0,
        label="score", session_names=session_names, area_label=None, ylabel="Score")

    # =====================================================================
    # SLIDING-WINDOW COHEN'S D (pooled, session-level) -- ACCURACY & SCORE
    # =====================================================================
    sliding_d_total = etf.sliding_window_effect_size(correct_acc_list, error_acc_list, centers_ref,
                                                   window_width=5, sd_type="total")
    sliding_d_within = etf.sliding_window_effect_size(correct_acc_list, error_acc_list, centers_ref,
                                                        window_width=5, sd_type="within")

    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
    ax_cmp.plot(sliding_d_total["window_centers"], sliding_d_total["d_values"], label="total SD (all 32 values)")
    ax_cmp.plot(sliding_d_within["window_centers"], sliding_d_within["d_values"], label="within-group pooled SD")
    ax_cmp.axhline(0, color="k", lw=1)
    ax_cmp.set_xlabel("Frame (window center)")
    ax_cmp.set_ylabel("Cohen's d")
    ax_cmp.legend()
    
    

    sliding_d_score = etf.sliding_window_effect_size(correct_score_list, error_score_list, centers_ref,
                                                       window_width=5, sd_type="total")
    etf.plot_sliding_window_effect_size(sliding_d_score, area_label="Score")

else:
    print("No sessions processed -- nothing to grand-average.")
    
    
    
    
# =====================================================================
# GRAND AVERAGE + STATS -- ACCURACY and SCORE (thesis figures)
# Copy this block unchanged into any area's script; edit AREA_TAG only.
# =====================================================================

AREA_TAG  = "allareas"          # "allareas" | "V1" | "V2" | "V4"
PLOT_KW   = dict(c_correct="#CC79A7", c_error="#8C8C8C", c_compare="#0072B2",
                 font_size=11, figsize_cm=(17, 11), xlim_ms=(-100, 280))
AREA_LBL  = None if AREA_TAG == "allareas" else AREA_TAG

if len(error_acc_list) > 0:

    # ---------------- ACCURACY ----------------
    sig_results = etf.run_group_significance_tests(
        correct_acc_list, error_acc_list, centers_ref, null_value=0.5)
    for comp in ('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'):
        n_sig = sig_results[comp]['sig_mask'].sum()
        n_valid = (~np.isnan(sig_results[comp]['p_raw'])).sum()
        print(f"[{AREA_TAG} ACC] {comp}: {n_sig} significant out of {n_valid} valid timepoints")

    fig, ax = etf.plot_correct_vs_error_thesis(
        correct_acc_list, error_acc_list, centers_ref, sig_results,
        ylabel="Decoding accuracy", chance=0.5, ylim=(0.4, 1.0), **PLOT_KW)

    window_result_acc = etf.test_window_average(
        correct_acc_list, error_acc_list, centers_ref, window=(41, 45), null_value=0.5,
        label="accuracy", session_names=session_names, area_label=AREA_LBL, ylabel="Accuracy")

    # ---------------- SCORE ----------------
    sig_score_results = etf.run_group_significance_tests(
        correct_score_list, error_score_list, centers_ref, null_value=0.0)
    for comp in ('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'):
        n_sig = sig_score_results[comp]['sig_mask'].sum()
        n_valid = (~np.isnan(sig_score_results[comp]['p_raw'])).sum()
        print(f"[{AREA_TAG} SCORE] {comp}: {n_sig} significant out of {n_valid} valid timepoints")

    fig2, ax2 = etf.plot_correct_vs_error_thesis(
        correct_score_list, error_score_list, centers_ref, sig_score_results,
        ylabel="Decision score (+ = true label)", chance=0.0, ylim=None, **PLOT_KW)

    window_result_score = etf.test_window_average(
        correct_score_list, error_score_list, centers_ref, window=(41, 45), null_value=0.0,
        label="score", session_names=session_names, area_label=AREA_LBL, ylabel="Score")

# keep all figures open until you're done looking
plt.show(block=True)


