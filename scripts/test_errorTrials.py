"""
test_errorTrials.py

Driver script for Aim 3, Part 2. Loops over sessions:
  1. Loads that session's already-trained sliding-window decoder (results)
  2. Processes that session's error trials against it
  3. Collects per-session results
  4. Builds a grand average across sessions
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from scripts.functions_scripts import errorTrials_Functions as etf
from scripts.functions_scripts.save_results import load_experiment, load_data_from_config

# =====================================================================
# CONFIG -- one entry per session
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
# PER-SESSION PIPELINE
# =====================================================================

error_acc_list = []
correct_acc_list = []
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
    )

    error_acc_list.append(acc_results["error_acc"])
    correct_acc_list.append(acc_results["correct_acc"])
    session_names.append(s["tag"])
    centers_ref = acc_results["centers"]

    per_session_results[s["tag"]] = {
        "results": results, "acc_results": acc_results, "score_results": score_results,
        "meta": meta, "X_error_roi": X_error_roi, "y_trials": y_trials,
    }

    # accuracy plot
    #fig, ax = etf.plot_accuracy_comparison(acc_results, session_tag=s["tag"])
    

    # score plot
    #fig, ax = etf.plot_fold_averaged_comparison(score_results, session_tag=s["tag"])
    

# =====================================================================
# GRAND AVERAGE ACROSS SESSIONS
# =====================================================================

if len(error_acc_list) > 0:
    grand_results = etf.grand_average_correct_and_error(correct_acc_list, error_acc_list, session_names, centers_ref)
    fig, ax = etf.plot_grand_average_correct_and_error(grand_results)
    #ax.set_xlim(0, 60)
else:
    print("No sessions processed -- nothing to grand-average.")

# keep all figures open until you're done looking
plt.show(block=True)