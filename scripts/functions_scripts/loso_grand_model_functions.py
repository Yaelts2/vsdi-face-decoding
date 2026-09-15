"""
loso_grand_model_functions.py

All functions for the leave-one-session-out (LOSO) grand SVM decoding model.
Kept separate from the main run script (loso_grand_model_main.py), which only
defines the session list and calls these.

Conventions matched to the existing single-session sliding-window pipeline:
    ZERO_FRAME = 27, FRAME_DURATION_MS = 10
    window_size=5, start_frame=1, stop_frame=100, step=1
    centers = start + window_size // 2
    FACE_LABEL = 1, NONFACE_LABEL = 0
    baseline z-score: subtract per-trial baseline mean, divide by a pooled std
    (zscore_dataset_pixelwise_trials pattern)

NEW for the grand model (per your instruction): the pooled std is computed
across all trials from all TRAINING sessions in a given LOSO fold (not per
session). It is recomputed fresh for each held-out session, using only the
15 training sessions -- the held-out session never contributes to the std
used to scale itself, since that would leak test-set information into
preprocessing. Flag if you actually want it computed once across all 16
sessions regardless of fold; happy to switch.
"""

import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC

ZERO_FRAME = 27
FRAME_DURATION_MS = 10.0
FACE_LABEL = 1
NONFACE_LABEL = 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_session_raw(face_file, nonface_file, data_dir):
    """
    Load one session's face and non-face condsXn matrices and combine them.

    face_file, nonface_file : str, filenames like 'condsXn1_030209a.npy'
    data_dir : str or Path, folder containing the .npy files

    Returns
    -------
    X : ndarray (pixels, frames, trials)
    y : ndarray (trials,) -- FACE_LABEL / NONFACE_LABEL
    """
    data_dir = Path(data_dir)
    X_face = np.load(data_dir / face_file)
    X_non = np.load(data_dir / nonface_file)

    n = min(X_face.shape[2], X_non.shape[2])
    X_face = X_face[:, :, :n]
    X_non = X_non[:, :, :n]

    X = np.concatenate([X_face, X_non], axis=2)
    y = np.array([FACE_LABEL] * n + [NONFACE_LABEL] * n)
    return X, y


def load_all_sessions(sessions, data_dir):
    """
    sessions : list of dict, each {'id': str, 'face_file': str, 'nonface_file': str}
    Returns list of dict, each {'id', 'X' (pixels,frames,trials), 'y' (trials,)}
    """
    loaded = []
    for sess in sessions:
        X, y = load_session_raw(sess['face_file'], sess['nonface_file'], data_dir)
        loaded.append({'id': sess['id'], 'X': X, 'y': y})
        print(f"  loaded {sess['id']}: X={X.shape}, n_face={int((y == FACE_LABEL).sum())}, "
              f"n_nonface={int((y == NONFACE_LABEL).sum())}")
    return loaded


# ---------------------------------------------------------------------------
# Pixel-wise z-score (pooled std) -- exact single-session function, verbatim
# ---------------------------------------------------------------------------
def zscore_dataset_pixelwise_trials(X, baseline_frames=(1, 24), eps: float = 1e-8, ddof: int = 0):
    """
    Updated Pixel-wise Z-score:
    1. Subtract trial-specific baseline mean from each trial.
    2. Calculate STD of those 'mean-centered' baseline segments pooled across trials.
    3. Divide by that pooled STD.

    Unmodified from the single-session pipeline. Used as-is when you want to
    z-score one session on its own. For the LOSO grand model, see
    baseline_center / pooled_std_across_sessions below, which reproduce this
    exact same math but pool the std across multiple sessions' trials.
    """
    X = np.asarray(X, dtype=float)
    pixels, frames, trials = X.shape

    start, end = baseline_frames
    baseline = X[:, start:end, :]
    mean_per_trial = baseline.mean(axis=1, keepdims=True)
    X_centered = X - mean_per_trial

    baseline_centered = X_centered[:, start:end, :]
    std_pooled = baseline_centered.std(axis=(1, 2), keepdims=True, ddof=ddof)

    X_z = X_centered / np.maximum(std_pooled, eps)
    return X_z, mean_per_trial, std_pooled


def baseline_center(X, baseline_frames=(1, 24)):
    """
    Step 1-2 of zscore_dataset_pixelwise_trials only: subtract each trial's
    own baseline mean (per pixel). Per-trial, so this doesn't leak
    information across sessions or folds -- safe to precompute once per
    session before the LOSO fold loop.

    X : (pixels, frames, trials)
    Returns X_centered : (pixels, frames, trials)
    """
    start, end = baseline_frames
    mean_per_trial = X[:, start:end, :].mean(axis=1, keepdims=True)
    return X - mean_per_trial


def pooled_std_across_sessions(centered_sessions, baseline_frames=(1, 24), eps=1e-8, ddof=0):
    """
    Step 3-4 of zscore_dataset_pixelwise_trials, but pooling the centered
    baseline across trials from MULTIPLE sessions instead of one session.

    This is mathematically identical to concatenating the raw trials from
    all these sessions along the trial axis and calling
    zscore_dataset_pixelwise_trials() once on the combined array -- just
    done from the already-centered per-session data so centering isn't
    redone on every fold.

    centered_sessions : list of ndarray (pixels, frames, trials), already
        baseline-centered (output of baseline_center)
    Returns std_pooled : (pixels, 1, 1)
    """
    start, end = baseline_frames
    baseline_chunks = [Xc[:, start:end, :] for Xc in centered_sessions]
    pooled = np.concatenate(baseline_chunks, axis=2)  # (pixels, n_baseline_frames, total_trials_all_sessions)
    std_pooled = pooled.std(axis=(1, 2), keepdims=True, ddof=ddof)
    return np.maximum(std_pooled, eps)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Frame-as-samples helpers -- local equivalents of your fe.frames_as_samples /
# cv.majority_vote_trial_predictions, since each window's samples here are
# individual FRAMES (not window-averaged pixels), same as your main
# sliding_window_decode_with_stats.
# ---------------------------------------------------------------------------
def frames_as_samples(X_win, y_trials, trial_axis=2, frame_axis=1, pixel_axis=0):
    """
    Flatten a (pixels, window_size, trials) window into frame-level samples:
    each frame of each trial becomes its own sample, carrying that trial's
    label, with a group id equal to the trial index for majority-vote
    aggregation afterward.

    Returns
    -------
    X_frames : (trials * window_size, pixels)
    y_frames : (trials * window_size,)
    groups   : (trials * window_size,) -- trial index per frame-sample
    """
    X_moved = np.moveaxis(X_win, [trial_axis, frame_axis, pixel_axis], [0, 1, 2])  # (trials, window, pixels)
    n_trials, n_frames_win, n_pixels = X_moved.shape
    X_frames = X_moved.reshape(n_trials * n_frames_win, n_pixels)
    y_frames = np.repeat(np.asarray(y_trials), n_frames_win)
    groups = np.repeat(np.arange(n_trials), n_frames_win)
    return X_frames, y_frames, groups


def majority_vote_trial_predictions(y_pred, y_frames, groups):
    """
    Collapse frame-level predictions to one prediction per trial by majority
    vote across that trial's frames within the window.
    """
    trial_ids = np.unique(groups)
    y_true_trial = np.empty(len(trial_ids), dtype=y_frames.dtype)
    y_pred_trial = np.empty(len(trial_ids), dtype=y_pred.dtype)
    for i, tid in enumerate(trial_ids):
        mask = groups == tid
        y_true_trial[i] = y_frames[mask][0]  # identical for every frame in a trial, by construction
        vals, counts = np.unique(y_pred[mask], return_counts=True)
        y_pred_trial[i] = vals[np.argmax(counts)]
    return y_true_trial, y_pred_trial


def window_bounds(n_frames, window_size=5, start_frame=1, stop_frame=100, step=1):
    """
    Exact window-loop bounds from sliding_window_decode_with_stats:
    last_start = min(n_frames - window_size, stop_frame - window_size);
    windows are [start:start+window_size) for start in
    range(start_frame, last_start+1, step).
    """
    stop_frame = min(int(stop_frame), n_frames)
    last_start = min(n_frames - window_size, stop_frame - window_size)
    if last_start < start_frame:
        raise ValueError("stop_frame is too early for the given start_frame/window_size")
    return list(range(start_frame, last_start + 1, step))


def centers_to_time_ms(centers, zero_frame=ZERO_FRAME, frame_duration_ms=FRAME_DURATION_MS):
    return (np.asarray(centers) - zero_frame) * frame_duration_ms


# ---------------------------------------------------------------------------
# LOSO grand model
# ---------------------------------------------------------------------------
def run_loso_grand_model(sessions_data, make_estimator, window_size=5, start_frame=1, stop_frame=100, step=1,
                          baseline_frames=(1, 24), verbose=True):
    """
    Leave-one-session-out grand model, matching sliding_window_decode_with_stats's
    convention: each window's samples are individual FRAMES within that
    window (not window-averaged pixels), with trial id as the group for
    majority-vote trial-level accuracy.

    The outer split here is session-level (LOSO) instead of GroupKFold within
    one session: for each held-out session, one classifier per window is
    trained on every frame from every trial in the other 15 sessions
    (pooled), then evaluated on the held-out session's frames -- this is the
    direct analogue of your "final_models" fit-on-everything step, just with
    the held-out session excluded from that "everything".

    Z-score: pooled std is computed ONCE across ALL trials from ALL sessions
    (matching zscore_dataset_pixelwise_trials exactly, just pooled over every
    session's trials instead of one session's), and reused as-is for every
    fold -- not recomputed per fold from training sessions only.

    sessions_data : list of dict {'id', 'X' (pixels,frames,trials), 'y' (trials,)}
    make_estimator : callable -> fresh estimator each fit, e.g.
        lambda: LinearSVC(C=1.0, max_iter=10000)

    Returns
    -------
    results : dict
        'session_ids'   : list[str]
        'centers'       : ndarray (n_windows,)
        'time_ms'       : ndarray (n_windows,)
        'frame_acc'     : ndarray (n_sessions, n_windows) -- held-out frame-level accuracy
        'trial_acc'     : ndarray (n_sessions, n_windows) -- held-out trial-level (majority vote) accuracy
        'weights'       : list[list[ndarray]] -- weights[fold][window] -> (n_pixels,), fit on pooled training frames
        'n_test_trials' : ndarray (n_sessions,)
        'params'        : dict of the window/baseline params used
    """
    session_ids = [s['id'] for s in sessions_data]
    n_sessions = len(sessions_data)
    n_frames_total = sessions_data[0]['X'].shape[1]

    starts = window_bounds(n_frames_total, window_size, start_frame, stop_frame, step)
    centers = np.array([s + window_size // 2 for s in starts])
    n_windows = len(starts)
    time_ms = centers_to_time_ms(centers)

    # Baseline-center once per session (per-trial, independent of fold)
    centered_raw = {sess['id']: baseline_center(sess['X'], baseline_frames) for sess in sessions_data}

    # Pooled std across ALL trials from ALL sessions (not per fold) -- same math as
    # zscore_dataset_pixelwise_trials, just pooling the centered baseline across every
    # session's trials at once, computed a single time up front.
    std_pooled = pooled_std_across_sessions(list(centered_raw.values()), baseline_frames)  # (pixels,1,1)

    if verbose:
        start_bl, end_bl = baseline_frames
        all_baseline_z = np.concatenate(
            [Xc[:, start_bl:end_bl, :] / std_pooled for Xc in centered_raw.values()], axis=2
        )
        print(f"sanity check (all-sessions baseline, z-scored): "
              f"mean={np.nanmean(all_baseline_z):.4f}, std={np.nanstd(all_baseline_z):.4f} "
              f"(should be ~0, ~1)")
        del all_baseline_z

    # z-score every session once with the global pooled std (same transform for train and test)
    z_sessions = {sid: Xc / std_pooled for sid, Xc in centered_raw.items()}

    results = {
        'session_ids': session_ids,
        'centers': centers,
        'time_ms': time_ms,
        'frame_acc': np.zeros((n_sessions, n_windows)),
        'trial_acc': np.zeros((n_sessions, n_windows)),
        'weights': [[None] * n_windows for _ in range(n_sessions)],
        'n_test_trials': np.array([len(s['y']) for s in sessions_data]),
        'params': {
            'window_size': window_size, 'start_frame': start_frame,
            'stop_frame': stop_frame, 'step': step, 'baseline_frames': baseline_frames,
        },
    }

    for fold_idx in range(n_sessions):
        held_out_id = session_ids[fold_idx]
        train_idx = [j for j in range(n_sessions) if j != fold_idx]

        if verbose:
            print(f"[fold {fold_idx + 1}/{n_sessions}] held-out session: {held_out_id}")

        z_train_sessions = [z_sessions[session_ids[j]] for j in train_idx]
        y_train_sessions = [sessions_data[j]['y'] for j in train_idx]
        z_test = z_sessions[held_out_id]
        y_test_trials = sessions_data[fold_idx]['y']

        for w, start in enumerate(starts):
            end = start + window_size

            X_frames_list, y_frames_list = [], []
            for Xz, y_trials in zip(z_train_sessions, y_train_sessions):
                X_win = Xz[:, start:end, :]
                Xf, yf, _ = frames_as_samples(X_win, y_trials, trial_axis=2, frame_axis=1, pixel_axis=0)
                X_frames_list.append(Xf)
                y_frames_list.append(yf)
            X_train = np.concatenate(X_frames_list, axis=0)
            y_train = np.concatenate(y_frames_list, axis=0)

            X_win_test = z_test[:, start:end, :]
            X_test_frames, y_test_frames, test_groups = frames_as_samples(
                X_win_test, y_test_trials, trial_axis=2, frame_axis=1, pixel_axis=0
            )

            clf = make_estimator()
            clf.fit(X_train, y_train)
            y_pred_frames = clf.predict(X_test_frames)

            frame_acc = float(np.mean(y_pred_frames == y_test_frames))
            y_true_trial, y_pred_trial = majority_vote_trial_predictions(y_pred_frames, y_test_frames, test_groups)
            trial_acc = float(np.mean(y_true_trial == y_pred_trial))

            results['frame_acc'][fold_idx, w] = frame_acc
            results['trial_acc'][fold_idx, w] = trial_acc
            results['weights'][fold_idx][w] = np.asarray(clf.coef_).ravel().astype(float)

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_loso_accuracy(results, metric='trial_acc', title=None, ax=None):
    """
    Mean +/- SEM accuracy across held-out sessions, per window, over time.
    Sessions are the replication unit (matches your Wilcoxon convention).

    metric : 'trial_acc' (majority-vote, default) or 'frame_acc'
    """
    import matplotlib.pyplot as plt

    time_ms = results['time_ms']
    acc = results[metric]  # (n_sessions, n_windows)
    mean_acc = acc.mean(axis=0)
    sem_acc = acc.std(axis=0, ddof=1) / np.sqrt(acc.shape[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    if title is None:
        title = f"LOSO grand model accuracy ({metric})"

    ax.plot(time_ms, mean_acc, color='black', lw=2, label='mean across sessions')
    ax.fill_between(time_ms, mean_acc - sem_acc, mean_acc + sem_acc, color='black', alpha=0.2, label='SEM')
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='chance')
    ax.axvline(0, color='red', ls='--', lw=1, label='stimulus onset')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Held-out session accuracy')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 1)
    return ax


def plot_loso_session_heatmap(results, metric='trial_acc', title=None, ax=None):
    """
    Heatmap: sessions x windows, accuracy. Lets you spot sessions that
    behave very differently from the rest of the grand model.

    metric : 'trial_acc' (majority-vote, default) or 'frame_acc'
    """
    import matplotlib.pyplot as plt

    time_ms = results['time_ms']
    acc = results[metric]
    if title is None:
        title = f"Per-session accuracy over time ({metric})"

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    im = ax.imshow(acc, aspect='auto', vmin=0, vmax=1, cmap='viridis',
                    extent=[time_ms[0], time_ms[-1], len(results['session_ids']), 0])
    ax.set_yticks(np.arange(len(results['session_ids'])) + 0.5)
    ax.set_yticklabels(results['session_ids'], fontsize=7)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='accuracy')
    return ax
