"""
benchmark_loso_runtime.py

Times ONE fold's ONE window fit+predict on your real data, then extrapolates
to the full 16-session x ~95-window LOSO run. Run this before the full
loso_grand_model_main.py so you know what you're committing to.
"""

import sys
import time
import numpy as np
from sklearn.svm import LinearSVC

sys.path.append(r"C:\project\vsdi-face-decoding\scripts\functions_scripts")
from loso_grand_model_functions import (
    load_all_sessions, baseline_center, pooled_std_across_sessions,
    frames_as_samples, window_bounds,
)

from loso_grand_model_main import SESSIONS, DATA_DIR, WINDOW_SIZE, START_FRAME, STOP_FRAME, STEP, BASELINE_FRAMES, C, MAX_ITER

print(f"Loading {len(SESSIONS)} sessions ...")
sessions_data = load_all_sessions(SESSIONS, DATA_DIR)
n_sessions = len(sessions_data)

centered = {s['id']: baseline_center(s['X'], BASELINE_FRAMES) for s in sessions_data}
std_pooled = pooled_std_across_sessions(list(centered.values()), BASELINE_FRAMES)
z_sessions = {sid: Xc / std_pooled for sid, Xc in centered.items()}

starts = window_bounds(sessions_data[0]['X'].shape[1], WINDOW_SIZE, START_FRAME, STOP_FRAME, STEP)
n_windows = len(starts)
print(f"n_sessions={n_sessions}, n_windows={n_windows}, total fits={n_sessions * n_windows}")

# --- Time ONE fold's ONE window ---
held_out_idx = 0
train_ids = [s['id'] for i, s in enumerate(sessions_data) if i != held_out_idx]
start = starts[len(starts) // 2]  # a middle window, representative of typical trial counts
end = start + WINDOW_SIZE

t0 = time.time()
X_list, y_list = [], []
for sid in train_ids:
    X_win = z_sessions[sid][:, start:end, :]
    y_trials = next(s['y'] for s in sessions_data if s['id'] == sid)
    Xf, yf, _ = frames_as_samples(X_win, y_trials)
    X_list.append(Xf)
    y_list.append(yf)
X_train = np.concatenate(X_list, axis=0)
y_train = np.concatenate(y_list, axis=0)
t1 = time.time()

clf = LinearSVC(C=C, max_iter=MAX_ITER)
clf.fit(X_train, y_train)
t2 = time.time()

X_test_win = z_sessions[sessions_data[held_out_idx]['id']][:, start:end, :]
Xf_test, yf_test, _ = frames_as_samples(X_test_win, sessions_data[held_out_idx]['y'])
clf.predict(Xf_test)
t3 = time.time()

build_time = t1 - t0
fit_time = t2 - t1
predict_time = t3 - t2
per_window_time = build_time + fit_time + predict_time

print(f"\nX_train shape: {X_train.shape}")
print(f"build data: {build_time:.2f}s | fit: {fit_time:.2f}s | predict: {predict_time:.2f}s")
print(f"-> ~{per_window_time:.2f}s per (fold, window)")

total_seconds = per_window_time * n_sessions * n_windows
print(f"\nEstimated total runtime: {total_seconds/60:.1f} minutes "
      f"({total_seconds/3600:.2f} hours) for all {n_sessions} folds x {n_windows} windows")
