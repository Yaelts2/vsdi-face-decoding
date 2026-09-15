"""
test_errorTrials_RT.py

Driver script for Aim 3, Part 2 -- RT-ALIGNED MODELS. Same pipeline/tests as
test_errorTrials_allareas.py / _V1.py / _V2.py, but:
  - error trials come from per-condition RT-aligned .npy files (combined via
    process_rt_aligned_trials_for_session), NOT an errorTrialsStruct .mat file
    for the actual DECODING input.
  - HOWEVER, each session's stimulus-aligned errorTrialsStruct .mat file
    (error_mat_path below) IS still needed, purely to compute the correct
    per-trial baseline (RT-aligned trials have no true pre-stimulus period of
    their own -- see normalize_with_external_baseline /
    load_stim_aligned_error_baseline in errorTrials_Functions.py for why).
  - trials are only 45 frames long (trimmed around the reaction time, not the
    full stimulus-locked length) -- so window/x-axis interpretation differs
    from your other three scripts. THE FIXED-WINDOW TEST BELOW USES A
    PLACEHOLDER RANGE (25-35, centered near frame ~30 where you noted the RT
    event falls) -- confirm/adjust this once you've looked at the grand-average
    and sliding-window plots for real.

Loops over sessions:
  1. Loads that session's already-trained RT-aligned sliding-window decoder
  2. Loads and combines that session's RT-aligned error trials (per-condition
     .npy files) against it, baselined using that same session's stimulus-
     aligned error .mat file (error_mat_path)
  3. Collects per-session results (accuracy, score)
  4. Builds grand averages across sessions (accuracy, score)
  5. Runs group-level significance tests (per-timepoint + windowed)
  6. Runs sliding-window Cohen's d (pooled, session-level) for both metrics

Results of the (slow) per-session loop are cached to disk. Set USE_CACHE=False
whenever you change anything UPSTREAM of the grand-average section.

NOTE: USE_CACHE is forced to False below for this run, since the baseline fix
in errorTrials_Functions.py (process_rt_aligned_trials_for_session now uses
each trial's stim-aligned baseline instead of the RT-aligned cut's own
frames 1-24) changes everything upstream of the grand-average section. Any
cache written by the OLD code is stale and must not be reused. Once you've
confirmed this run looks right, you can flip USE_CACHE back to True for
faster reruns of just the grand-average/plotting section.
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

CACHE_PATH = Path(r"C:\project\vsdi-face-decoding\results\error_trials_cache_RT.pkl")
USE_CACHE = False      # forced False -- baseline fix changes everything upstream; see note above

RT_DATA_BASE_DIR = r"C:\project\vsdi-face-decoding\data\processed\RTaligend"
ERROR_MAT_BASE_DIR = r"C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\error trials"

MATCH_CORRECT_TO_ERROR = True  # if True, subsample correct trials to match the number of error trials
SUBSAMPLE_SEED = 42

# *** PLACEHOLDER -- adjust once you've seen the actual plots ***
# Frame range for the fixed-window test. You noted the RT event falls around
# frame ~30 in this alignment; this window straddles that as a starting guess.
FIXED_WINDOW = (25, 35)

# =====================================================================
# CONFIG -- one entry per session (RT-aligned models)
# session_prefix = tag with the trailing face/nonface condition digits
# stripped off (e.g. "110209a15" -> "110209a"), used to build the RT .npy
# file paths: RTaligned_<session_prefix><cond>_error.npy
#
# error_mat_path points at the STIMULUS-ALIGNED errorTrialsStruct .mat file
# for this same session/condition-pair -- same naming convention as your
# other test_errorTrials_*.py scripts: errorTrialsData_<tag>.mat under
# ERROR_MAT_BASE_DIR\<session_prefix>\. Used ONLY to compute the correct
# per-trial baseline for the RT-aligned data (see module docstring above).
# =====================================================================

sessions = [
    {
        "tag": "030209a15", "session_prefix": "030209a", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209a15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-30-42",
    },
    {
        "tag": "030209a24", "session_prefix": "030209a", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209a24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-34-09",
    },
    {
        "tag": "030209c15", "session_prefix": "030209c", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209c15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-38-30",
    },
    {
        "tag": "030209c24", "session_prefix": "030209c", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209c24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-42-27",
    },
    {
        "tag": "030209e15", "session_prefix": "030209e", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209e15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-46-38",
    },
    {
        "tag": "030209e24", "session_prefix": "030209e", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209e24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-50-56",
    },
    {
        "tag": "030209f15", "session_prefix": "030209f", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209f15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-55-03",
    },
    {
        "tag": "030209f24", "session_prefix": "030209f", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__030209f24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-58-27",
    },
    {
        "tag": "110209a15", "session_prefix": "110209a", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned_110209a15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_12-53-51",
    },
    {
        "tag": "110209a24", "session_prefix": "110209a", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned_110209a24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_12-58-45",
    },
    {
        "tag": "110209b15", "session_prefix": "110209b", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned_110209b15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-03-10",
    },
    {
        "tag": "110209b24", "session_prefix": "110209b", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned_110209b24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-08-15",
    },
    {
        "tag": "110209c15", "session_prefix": "110209c", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned_110209c15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-12-02",
    },
    {
        "tag": "110209c24", "session_prefix": "110209c", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned_110209c24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-15-40",
    },
    {
        "tag": "110209d15", "session_prefix": "110209d", "face_cond": 1, "nonface_cond": 5,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned_110209d15_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-18-36",
    },
    {
        "tag": "110209d24", "session_prefix": "110209d", "face_cond": 2, "nonface_cond": 4,
        "run_dir": r"C:\project\vsdi-face-decoding\results\RTalignedModels\sliding_window_RTaligned__110209d24_RTaligned_frame1-40__SVM_10foldCV____2026-09-03_13-27-19",
    },
]


# Sanity check: catch unresolved placeholder run_dirs loudly instead of letting
# them fail deep inside load_experiment with a confusing file-not-found error.
for s in sessions:
    if s["run_dir"].startswith("REPLACE_ME"):
        raise ValueError(
            f"Session '{s['tag']}' has an unresolved run_dir placeholder -- "
            f"see the note above it in the sessions list. Fix before running."
        )

# Fill in error_mat_path for every session using the same convention as your
# other test_errorTrials_*.py scripts: ERROR_MAT_BASE_DIR\<session_prefix>\errorTrialsData_<tag>.mat
for s in sessions:
    s["error_mat_path"] = str(Path(ERROR_MAT_BASE_DIR) / s["session_prefix"] / f"errorTrialsData_{s['tag']}.mat")

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

        acc_results, score_results, meta, X_error_roi = etf.process_rt_aligned_trials_for_session(
            session_prefix=s["session_prefix"],
            face_cond=s["face_cond"],
            nonface_cond=s["nonface_cond"],
            base_dir=RT_DATA_BASE_DIR,
            results=results,
            ROI_mask_path=ROI_mask_path,
            y_trials=y_trials,
            error_mat_path=s["error_mat_path"],
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
# SANITY CHECK: print the actual window-center range for this RT-aligned
# decoder before trusting any fixed-window choice above.
# =====================================================================
if centers_ref is not None:
    print(f"\ncenters_ref range for RT-aligned models: {centers_ref.min()} to {centers_ref.max()} "
          f"({len(centers_ref)} windows)")
    print(f"FIXED_WINDOW is currently set to {FIXED_WINDOW} -- make sure this actually falls "
          f"within the range above before trusting test_window_average's output.")

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
    ax.set_title(ax.get_title() + " (RT-aligned)")

    sig_results = etf.run_group_significance_tests(correct_acc_list, error_acc_list, centers_ref, null_value=0.5)

    for comp in ('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'):
        n_sig = sig_results[comp]['sig_mask'].sum()
        n_valid = (~np.isnan(sig_results[comp]['p_raw'])).sum()
        print(f"[ACC] {comp}: {n_sig} significant out of {n_valid} valid timepoints")

    etf.add_significance_markers(ax, sig_results, comparisons=('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'))

    # ---- windowed test (PLACEHOLDER window -- see FIXED_WINDOW above) ----
    window_result_acc = etf.test_window_average(
        correct_acc_list, error_acc_list, centers_ref, window=FIXED_WINDOW, null_value=0.5,
        label="accuracy", session_names=session_names, area_label="RT-aligned", ylabel="Accuracy")

    # =====================================================================
    # GRAND AVERAGE + STATS -- SCORE
    # =====================================================================
    grand_score_results = etf.grand_average_metric(
        correct_score_list, error_score_list, session_names, centers_ref, metric_name="score")
    fig2, ax2 = etf.plot_grand_average_metric(
        grand_score_results, ylabel="Mean score (+ = matches own true label)",
        title=f"Grand average across {grand_score_results['n_sessions']} sessions: correct vs. error label-consistent score (RT-aligned)",
        ref_line=0, ref_label="Decision boundary (0)")

    sig_score_results = etf.run_group_significance_tests(
        correct_score_list, error_score_list, centers_ref, null_value=0.0)

    for comp in ('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'):
        n_sig = sig_score_results[comp]['sig_mask'].sum()
        n_valid = (~np.isnan(sig_score_results[comp]['p_raw'])).sum()
        print(f"[SCORE] {comp}: {n_sig} significant out of {n_valid} valid timepoints")

    etf.add_significance_markers(ax2, sig_score_results,
        comparisons=('correct_vs_error', 'correct_vs_chance', 'error_vs_chance'))

    # ---- windowed test (PLACEHOLDER window -- see FIXED_WINDOW above) ----
    window_result_score = etf.test_window_average(
        correct_score_list, error_score_list, centers_ref, window=FIXED_WINDOW, null_value=0.0,
        label="score", session_names=session_names, area_label="RT-aligned", ylabel="Score")

    # =====================================================================
    # SLIDING-WINDOW COHEN'S D (pooled, session-level) -- ACCURACY & SCORE
    # =====================================================================
    sliding_d_acc = etf.sliding_window_effect_size(correct_acc_list, error_acc_list, centers_ref,
                                                     window_width=16, method="pooled")
    etf.plot_sliding_window_effect_size(sliding_d_acc, area_label="Accuracy (RT-aligned)")

    sliding_d_score = etf.sliding_window_effect_size(correct_score_list, error_score_list, centers_ref,
                                                       window_width=16, method="pooled")
    etf.plot_sliding_window_effect_size(sliding_d_score, area_label="Score (RT-aligned)")

else:
    print("No sessions processed -- nothing to grand-average.")

# keep all figures open until you're done looking
plt.show(block=True)