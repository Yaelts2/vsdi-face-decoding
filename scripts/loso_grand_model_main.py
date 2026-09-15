"""
loso_grand_model_main.py

Run the leave-one-session-out grand SVM decoding model.
All functions live in loso_grand_model_functions.py -- this script just
defines the session list and calls them.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# loso_grand_model_functions.py now lives in a separate folder -- add it to the path so the import below finds it
sys.path.append(r"C:\project\vsdi-face-decoding\scripts\functions_scripts")

from loso_grand_model_functions import (
    load_all_sessions,
    run_loso_grand_model,
    plot_loso_accuracy,
    plot_loso_session_heatmap,
)

# ---------------------------------------------------------------------------
# Session list -- fill in with your 16 sessions.
# face_file / nonface_file are the .npy filenames inside DATA_DIR.
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\project\vsdi-face-decoding\data\processed\condsXn"

SESSIONS = [
    {'id': '110209a15', 'face_file': 'condsXn1_110209a.npy', 'nonface_file': 'condsXn5_110209a.npy'},
    {'id': '110209a24', 'face_file': 'condsXn2_110209a.npy', 'nonface_file': 'condsXn4_110209a.npy'},
    {'id': '110209b15', 'face_file': 'condsXn1_110209b.npy', 'nonface_file': 'condsXn5_110209b.npy'},
    {'id': '110209b24', 'face_file': 'condsXn2_110209b.npy', 'nonface_file': 'condsXn4_110209b.npy'},
    {'id': '110209c15', 'face_file': 'condsXn1_110209c.npy', 'nonface_file': 'condsXn5_110209c.npy'},
    {'id': '110209c24', 'face_file': 'condsXn2_110209c.npy', 'nonface_file': 'condsXn4_110209c.npy'},
    {'id': '110209d15', 'face_file': 'condsXn1_110209d.npy', 'nonface_file': 'condsXn5_110209d.npy'},
    {'id': '110209d24', 'face_file': 'condsXn2_110209d.npy', 'nonface_file': 'condsXn4_110209d.npy'},

    {'id': '030209a15', 'face_file': 'condsXn1_030209a.npy', 'nonface_file': 'condsXn5_030209a.npy'},
    {'id': '030209a24', 'face_file': 'condsXn2_030209a.npy', 'nonface_file': 'condsXn4_030209a.npy'},
    {'id': '030209c15', 'face_file': 'condsXn1_030209c.npy', 'nonface_file': 'condsXn5_030209c.npy'},
    {'id': '030209c24', 'face_file': 'condsXn2_030209c.npy', 'nonface_file': 'condsXn4_030209c.npy'},
    {'id': '030209e15', 'face_file': 'condsXn1_030209e.npy', 'nonface_file': 'condsXn5_030209e.npy'},
    {'id': '030209e24', 'face_file': 'condsXn2_030209e.npy', 'nonface_file': 'condsXn4_030209e.npy'},
    {'id': '030209f15', 'face_file': 'condsXn1_030209f.npy', 'nonface_file': 'condsXn5_030209f.npy'},
    {'id': '030209f24', 'face_file': 'condsXn2_030209f.npy', 'nonface_file': 'condsXn4_030209f.npy'},
    
]

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
WINDOW_SIZE = 5
START_FRAME = 1
STOP_FRAME = 100
STEP = 1
BASELINE_FRAMES = (1, 24)
C = 0.0001
MAX_ITER = 10000

def make_estimator():
    return LinearSVC(C=C, max_iter=MAX_ITER)

if __name__ == "__main__":
    print(f"Loading {len(SESSIONS)} sessions from {DATA_DIR} ...")
    sessions_data = load_all_sessions(SESSIONS, DATA_DIR)

    print("\nRunning LOSO grand model ...")
    results = run_loso_grand_model(
        sessions_data,
        make_estimator,
        window_size=WINDOW_SIZE,
        start_frame=START_FRAME,
        stop_frame=STOP_FRAME,
        step=STEP,
        baseline_frames=BASELINE_FRAMES,
    )

    out_path = "loso_grand_model_results.npz"
    np.savez(out_path, **{k: v for k, v in results.items() if k not in ('weights', 'params')},
             weights=np.array(results['weights'], dtype=object))
    print(f"\nSaved results to {out_path}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    plot_loso_accuracy(results, metric='trial_acc', ax=axes[0])
    plot_loso_session_heatmap(results, metric='trial_acc', ax=axes[1])
    plt.tight_layout()
    plt.savefig("loso_grand_model_figures.png", dpi=150)
    print("Saved figures to loso_grand_model_figures.png")
    plt.show()