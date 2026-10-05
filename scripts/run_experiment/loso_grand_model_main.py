"""
loso_grand_model_main.py

Train the nested leave-one-face-out grand SVM model and save it.
All functions live in loso_grand_model_functions.py - this script only defines the datasets and
parameters and calls them.

Structure (see the functions file for details):
    outer: leave out all datasets that show the SAME face image (12 faces -> 12 groups)
    inner: stratified 10-fold CV over the mixed trials of the remaining datasets -> 10 models
    every model is tested on its own held-out 10% AND on the left-out group (10 accuracies per group)
"""

import sys
from pathlib import Path

from sklearn.svm import LinearSVC

# folder that holds loso_grand_model_functions.py
sys.path.append(r"C:\project\vsdi-face-decoding\scripts\functions_scripts")

from loso_grand_model_functions import (
    load_all_datasets,
    build_leave_out_groups,
    estimate_runtime,
    run_nested_loso,
    save_results,
)

# ---------------------------------------------------------------------------
# Datasets. face_id = the face image of the dataset (from the database table);
# scramble = 'shuf' (segment scrambling) or 'ps' (phase scrambling).
# Datasets with the same face_id are always left out together.
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\project\vsdi-face-decoding\data\processed\condsXn"
RESULTS_DIR = Path(r"C:\project\vsdi-face-decoding\results")
OUT_NAME = "loso_nested_grand_model_results.npz"

SESSIONS = [
    # 11/2/2009
    {'id': '110209a15', 'face_id': 10, 'scramble': 'shuf', 'face_file': 'condsXn1_110209a.npy', 'nonface_file': 'condsXn5_110209a.npy'},
    {'id': '110209a24', 'face_id': 15, 'scramble': 'shuf', 'face_file': 'condsXn2_110209a.npy', 'nonface_file': 'condsXn4_110209a.npy'},
    {'id': '110209b15', 'face_id': 17, 'scramble': 'ps',   'face_file': 'condsXn1_110209b.npy', 'nonface_file': 'condsXn5_110209b.npy'},
    {'id': '110209b24', 'face_id': 5,  'scramble': 'ps',   'face_file': 'condsXn2_110209b.npy', 'nonface_file': 'condsXn4_110209b.npy'},
    {'id': '110209c15', 'face_id': 4,  'scramble': 'shuf', 'face_file': 'condsXn1_110209c.npy', 'nonface_file': 'condsXn5_110209c.npy'},
    {'id': '110209c24', 'face_id': 6,  'scramble': 'shuf', 'face_file': 'condsXn2_110209c.npy', 'nonface_file': 'condsXn4_110209c.npy'},
    {'id': '110209d15', 'face_id': 3,  'scramble': 'ps',   'face_file': 'condsXn1_110209d.npy', 'nonface_file': 'condsXn5_110209d.npy'},
    {'id': '110209d24', 'face_id': 7,  'scramble': 'ps',   'face_file': 'condsXn2_110209d.npy', 'nonface_file': 'condsXn4_110209d.npy'},
    # 3/2/2009
    {'id': '030209a15', 'face_id': 2,  'scramble': 'shuf', 'face_file': 'condsXn1_030209a.npy', 'nonface_file': 'condsXn5_030209a.npy'},
    {'id': '030209a24', 'face_id': 12, 'scramble': 'shuf', 'face_file': 'condsXn2_030209a.npy', 'nonface_file': 'condsXn4_030209a.npy'},
    {'id': '030209c15', 'face_id': 8,  'scramble': 'ps',   'face_file': 'condsXn1_030209c.npy', 'nonface_file': 'condsXn5_030209c.npy'},
    {'id': '030209c24', 'face_id': 18, 'scramble': 'ps',   'face_file': 'condsXn2_030209c.npy', 'nonface_file': 'condsXn4_030209c.npy'},
    {'id': '030209e15', 'face_id': 10, 'scramble': 'ps',   'face_file': 'condsXn1_030209e.npy', 'nonface_file': 'condsXn5_030209e.npy'},
    {'id': '030209e24', 'face_id': 15, 'scramble': 'ps',   'face_file': 'condsXn2_030209e.npy', 'nonface_file': 'condsXn4_030209e.npy'},
    {'id': '030209f15', 'face_id': 8,  'scramble': 'shuf', 'face_file': 'condsXn1_030209f.npy', 'nonface_file': 'condsXn5_030209f.npy'},
    {'id': '030209f24', 'face_id': 18, 'scramble': 'shuf', 'face_file': 'condsXn2_030209f.npy', 'nonface_file': 'condsXn4_030209f.npy'},
]

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
WINDOW_SIZE = 5
START_FRAME = 1
STOP_FRAME = 100
STEP = 1                      # use STEP = 10 for a ~10x faster dry run
BASELINE_FRAMES = (1, 24)
C = 0.0001
MAX_ITER = 10000
N_FOLDS = 10                  # inner stratified K-fold over the mixed trials of the remaining datasets
SEED = 0                      # inner fold assignment (fold split of group g uses SEED + g)
N_JOBS = 4                    # threads fitting the 10 fold models of a window in parallel (1 = serial)
RESUME = True                 # skip groups already finished in the checkpoint folder (same parameters only)
CHECKPOINT_DIR = RESULTS_DIR / "loso_nested_checkpoints"
EXTRA_PARAMS = {'baseline_frames': list(BASELINE_FRAMES),
                'z_score': 'baseline-centered per trial; std pooled over all trials of all datasets'}

ESTIMATE_RUNTIME = True       # time 2 real windows and print the expected total run time before the long run
ESTIMATE_ONLY = True         # True = stop right after the estimate (nothing is trained or saved)


def make_estimator():
    return LinearSVC(C=C, max_iter=MAX_ITER, random_state=0)


if __name__ == "__main__":
    groups = build_leave_out_groups(SESSIONS)
    print(f"{len(SESSIONS)} datasets, {len(groups)} faces -> {len(groups)} left-out groups:")
    for g in groups:
        print(f"   {g['name']:<9}: {', '.join(SESSIONS[i]['id'] for i in g['dataset_idx'])}")

    print(f"\nLoading datasets from {DATA_DIR} ...")
    datasets, std_pooled = load_all_datasets(SESSIONS, DATA_DIR, baseline_frames=BASELINE_FRAMES,
                                             stop_frame=STOP_FRAME)

    if ESTIMATE_RUNTIME:
        print()
        estimate_runtime(
            datasets, groups, make_estimator,
            window_size=WINDOW_SIZE, start_frame=START_FRAME, stop_frame=STOP_FRAME, step=STEP,
            n_folds=N_FOLDS, seed=SEED, n_jobs=N_JOBS,
            checkpoint_dir=CHECKPOINT_DIR if RESUME else None, extra_params=EXTRA_PARAMS,
        )
        if ESTIMATE_ONLY:
            sys.exit(0)

    print("\nRunning the nested leave-one-face-out model ...")
    results = run_nested_loso(
        datasets, groups, make_estimator,
        window_size=WINDOW_SIZE, start_frame=START_FRAME, stop_frame=STOP_FRAME, step=STEP,
        n_folds=N_FOLDS, seed=SEED, n_jobs=N_JOBS,
        checkpoint_dir=CHECKPOINT_DIR if RESUME else None, extra_params=EXTRA_PARAMS,
    )

    out_path = save_results(results, RESULTS_DIR / OUT_NAME)
    print(f"\nSaved results to {out_path}")
    print("Open it with loso_grand_model_figures_main.py")