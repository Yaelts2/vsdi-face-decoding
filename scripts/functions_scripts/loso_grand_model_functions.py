"""
loso_grand_model_functions.py

All functions for the NESTED leave-one-face-out grand SVM decoding model
(VSDI, face vs non-face). Used by two scripts:
    loso_grand_model_main.py          - defines sessions/parameters, trains, saves the .npz
    loso_grand_model_figures_main.py  - opens the saved .npz, prints stats, makes the figures

Model structure
---------------
Outer loop (one iteration = one LEFT-OUT GROUP):
    All datasets that show the SAME face image are taken out together (so the model never
    trains on a face it is later tested on). 16 datasets / 12 faces -> 12 groups.
Inner loop:
    All remaining datasets are pooled and their TRIALS are mixed. A stratified 10-fold CV
    over the pooled trials gives 10 models (~10% of the pooled trials left out per fold,
    not one dataset per fold). Trials are split as units, so the frames of a trial never
    straddle train and test.
Each of the 10 models, in every sliding window, is tested on
    (a) its own held-out ~10% of the pooled trials     -> inner_* (within-pool accuracy)
    (b) every dataset of the left-out group             -> ho_*    (cross-dataset accuracy)
so every left-out group gets 10 accuracies per window.

Decoding convention (same as sliding_window_decode_with_stats)
--------------------------------------------------------------
window_size=5, start_frame=1, stop_frame=100, step=1; centers = start + window_size // 2;
each FRAME of a window is its own sample (trial = group); trial-level accuracy is the
majority vote over the frames of a trial; one classifier per window.

Z-score (same maths as zscore_dataset_pixelwise_trials)
-------------------------------------------------------
Per-trial baseline mean subtraction, then ONE std per pixel pooled across the baseline
frames of ALL trials of ALL datasets (computed once, used for every fold). The std is
accumulated dataset by dataset (sum / sum of squares in float64) instead of concatenating
all datasets, which gives the same numbers with a fraction of the memory.

Statistics
----------
The replication unit is the LEFT-OUT GROUP (12), never the 10 folds: the 10 models of a group
share ~80% of their training trials and are tested on the same left-out trials, so they are
not independent. Per group the 10 fold accuracies are averaged first. Group accuracy is the
trial-weighted mean over the datasets of the group (= accuracy on the pooled trials of the
group). Two-tailed Wilcoxon signed-rank vs chance, BH-FDR across post-stimulus windows.
Within-pool (inner) accuracies of different groups come from heavily overlapping pools, so
they are shown descriptively and are not tested.
"""

import datetime
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold

ZERO_FRAME = 27
FRAME_DURATION_MS = 10.0
FACE_LABEL = 1
NONFACE_LABEL = 0


# ===========================================================================
# 1. Loading and z-scoring
# ===========================================================================
def zscore_dataset_pixelwise_trials(X, baseline_frames=(1, 24), eps: float = 1e-8, ddof: int = 0):
    """
    Updated Pixel-wise Z-score:
    1. Subtract trial-specific baseline mean from each trial.
    2. Calculate STD of those 'mean-centered' baseline segments pooled across trials.
    3. Divide by that pooled STD.

    Verbatim single-array reference function (kept for checking and for z-scoring one
    dataset on its own). load_all_datasets() below does the same maths with the std pooled
    over all trials of all datasets, without building one huge array.
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


def load_session_raw(face_file, nonface_file, data_dir, stop_frame=100, dtype=np.float32):
    """
    Load one dataset's face and non-face condsXn matrices (pixels, 256, trials) and combine
    them (truncated to the same number of trials; FACE_LABEL=1, NONFACE_LABEL=0).
    Only frames [0, stop_frame) are read - nothing after stop_frame is ever used.

    Returns X : (pixels, stop_frame, trials) in `dtype`, y : (trials,) int
    """
    data_dir = Path(data_dir)
    X_face = np.load(data_dir / face_file, mmap_mode='r')
    X_non = np.load(data_dir / nonface_file, mmap_mode='r')
    if X_face.shape[:2] != X_non.shape[:2]:
        raise ValueError(f"{face_file} {X_face.shape} and {nonface_file} {X_non.shape} "
                         f"differ in pixels/frames")
    if X_face.shape[1] < stop_frame:
        raise ValueError(f"{face_file} has only {X_face.shape[1]} frames (< stop_frame={stop_frame})")

    n = min(X_face.shape[2], X_non.shape[2])
    X = np.empty((X_face.shape[0], stop_frame, 2 * n), dtype=dtype)
    X[:, :, :n] = X_face[:, :stop_frame, :n]
    X[:, :, n:] = X_non[:, :stop_frame, :n]
    y = np.array([FACE_LABEL] * n + [NONFACE_LABEL] * n, dtype=int)
    return X, y


def baseline_center_inplace(X, baseline_frames=(1, 24)):
    """Subtract each trial's own baseline mean (per pixel), in place. X: (pixels, frames, trials)."""
    start, end = baseline_frames
    X -= X[:, start:end, :].mean(axis=1, keepdims=True, dtype=np.float64)
    return X


def load_all_datasets(sessions, data_dir, baseline_frames=(1, 24), stop_frame=100,
                      eps=1e-8, ddof=0, dtype=np.float32, verbose=True):
    """
    Load every dataset, subtract each trial's baseline mean, and divide by ONE std per pixel
    pooled across the baseline frames of ALL trials of ALL datasets.

    sessions : list of dict {'id', 'face_id', 'scramble', 'face_file', 'nonface_file'}
    Returns
    -------
    datasets : list of dict {'id', 'face_id', 'scramble',
                             'X' : (trials, frames, pixels) z-scored, C-contiguous, `dtype`,
                             'y' : (trials,)}
    std_pooled : (pixels,) float64 - the pooled std that was used (before the eps floor)
    """
    start, end = baseline_frames
    centered = []
    s1 = s2 = None
    n_total = 0

    for sess in sessions:
        X, y = load_session_raw(sess['face_file'], sess['nonface_file'], data_dir, stop_frame, dtype)
        if not np.isfinite(X).all():
            raise ValueError(f"Dataset {sess['id']} contains NaN/Inf values")
        baseline_center_inplace(X, baseline_frames)

        B = X[:, start:end, :].astype(np.float64)
        if s1 is None:
            s1 = np.zeros(X.shape[0])
            s2 = np.zeros(X.shape[0])
        s1 += B.sum(axis=(1, 2))
        s2 += (B ** 2).sum(axis=(1, 2))
        n_total += B.shape[1] * B.shape[2]
        del B

        centered.append((sess, X, y))
        if verbose:
            print(f"  loaded {sess['id']} (face {sess.get('face_id')}, {sess.get('scramble', '')}): "
                  f"{X.shape[0]} pixels, {X.shape[1]} frames, "
                  f"{int((y == FACE_LABEL).sum())} face + {int((y == NONFACE_LABEL).sum())} non-face trials")

    mean = s1 / n_total
    var = (s2 - n_total * mean ** 2) / (n_total - ddof)
    std_pooled = np.sqrt(np.maximum(var, 0.0))
    divisor = np.maximum(std_pooled, eps).astype(dtype)[:, None, None]

    datasets = []
    z1 = z2 = 0.0
    zn = 0
    for i in range(len(centered)):
        sess, X, y = centered[i]
        centered[i] = None
        X /= divisor

        B = X[:, start:end, :].astype(np.float64)
        z1 += float(B.sum())
        z2 += float((B ** 2).sum())
        zn += B.size
        del B

        Xt = np.ascontiguousarray(X.transpose(2, 1, 0))      # (trials, frames, pixels)
        del X
        datasets.append({'id': sess['id'], 'face_id': sess.get('face_id'),
                         'scramble': sess.get('scramble', ''), 'X': Xt, 'y': y})

    if verbose:
        zmean = z1 / zn
        zstd = float(np.sqrt(z2 / zn - zmean ** 2))
        print(f"sanity check (all-datasets baseline, z-scored): mean={float(zmean):.4f}, "
              f"std={zstd:.4f} (should be ~0, ~1); {n_total // (end - start)} trials pooled for the std")
    return datasets, std_pooled


# ===========================================================================
# 2. Groups, windows, frame samples
# ===========================================================================
def build_leave_out_groups(sessions):
    """
    One group per face image: all datasets showing the same face_id are left out together.
    Returns list of dict {'name', 'face_id', 'dataset_idx'} sorted by face_id.
    """
    face_ids = sorted({s['face_id'] for s in sessions})
    groups = []
    for fid in face_ids:
        idx = [i for i, s in enumerate(sessions) if s['face_id'] == fid]
        groups.append({'name': f"face {fid}", 'face_id': fid, 'dataset_idx': idx})
    return groups


def window_bounds(n_frames, window_size=5, start_frame=1, stop_frame=100, step=1):
    """
    Exact window-loop bounds from sliding_window_decode_with_stats:
    last_start = min(n_frames - window_size, stop_frame - window_size);
    windows are [start, start+window_size) for start in range(start_frame, last_start+1, step).
    """
    stop_frame = min(int(stop_frame), n_frames)
    last_start = min(n_frames - window_size, stop_frame - window_size)
    if last_start < start_frame:
        raise ValueError("stop_frame is too early for the given start_frame/window_size")
    return list(range(start_frame, last_start + 1, step))


def centers_to_time_ms(centers, zero_frame=ZERO_FRAME, frame_duration_ms=FRAME_DURATION_MS):
    return (np.asarray(centers) - zero_frame) * frame_duration_ms


def window_frame_samples(X_tfp, start, end):
    """
    Frames-as-samples for one window. X_tfp : (trials, frames, pixels).
    Returns (trials * (end-start), pixels) float64, trial-major: all frames of trial 0,
    then all frames of trial 1, ... (same ordering as fe.frames_as_samples).
    """
    n_tr, _, n_pix = X_tfp.shape
    return np.array(X_tfp[:, start:end, :], dtype=np.float64, order='C').reshape(n_tr * (end - start), n_pix)


def majority_vote_trial(y_pred_frames, window_size):
    """
    One prediction per trial = majority over the window_size frames of the trial
    (frames are trial-major and every trial has exactly window_size frames).
    Labels are 0/1; use an odd window_size so there are no ties (a tie would give 0).
    """
    votes = np.asarray(y_pred_frames).reshape(-1, window_size).mean(axis=1)
    return (votes > 0.5).astype(int)


def make_inner_folds(pool_y_trials, n_folds, seed, window_size):
    """
    Stratified (face/non-face) K-fold over the pooled TRIALS of all non-left-out datasets.
    Returns a list of dicts with the trial indices and the matching frame-sample rows.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ar = np.arange(window_size)
    folds = []
    for tr, te in skf.split(np.zeros(len(pool_y_trials)), pool_y_trials):
        folds.append({
            'train_trials': tr,
            'test_trials': te,
            'train_rows': (tr[:, None] * window_size + ar).ravel(),
            'test_rows': (te[:, None] * window_size + ar).ravel(),
        })
    return folds


# ===========================================================================
# 3. Nested leave-one-face-out
# ===========================================================================
def _fit_eval_fold(make_estimator, X_pool, y_pool_frames, train_rows, test_rows, test_sets, window_size):
    """
    Fit one inner-fold model; evaluate on its own held-out rows and on each left-out dataset.
    test_sets : list of (X_frames, y_frames) for the left-out datasets.
    Returns (inner_frame_acc, inner_trial_acc, [(frame_acc, trial_acc) per left-out dataset], weights)
    """
    clf = make_estimator()
    clf.fit(X_pool[train_rows], y_pool_frames[train_rows])

    yt = y_pool_frames[test_rows]
    yp = clf.predict(X_pool[test_rows])
    inner_frame = float(np.mean(yp == yt))
    inner_trial = float(np.mean(majority_vote_trial(yp, window_size) == yt.reshape(-1, window_size)[:, 0]))

    ho = []
    for X_te, y_te in test_sets:
        yp = clf.predict(X_te)
        frame = float(np.mean(yp == y_te))
        trial = float(np.mean(majority_vote_trial(yp, window_size) == y_te.reshape(-1, window_size)[:, 0]))
        ho.append((frame, trial))

    return inner_frame, inner_trial, ho, np.asarray(clf.coef_, dtype=float).ravel()


def _group_signature(params, group, datasets, test_idx, pool_idx, make_estimator):
    return json.dumps({
        'params': params,
        'group': group['name'],
        'test': [datasets[i]['id'] for i in test_idx],
        'pool': [datasets[i]['id'] for i in pool_idx],
        'n_trials': [int(len(datasets[i]['y'])) for i in range(len(datasets))],
        'estimator': repr(make_estimator()),
    }, sort_keys=True, default=str)


def _save_checkpoint(path, signature, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, signature=np.array(signature), **arrays)
    tmp.replace(path)


def _load_checkpoint(path, signature):
    try:
        with np.load(path, allow_pickle=False) as d:
            if str(d['signature']) != signature:
                return None
            return {k: d[k] for k in d.files if k != 'signature'}
    except Exception:
        return None


def _check_only_groups(groups, only_groups):
    if only_groups is None:
        return None
    names = [g['name'] for g in groups]
    bad = [n for n in only_groups if n not in names]
    if bad:
        raise ValueError(f"only_groups {bad} not found; available groups: {names}")
    return set(only_groups)


def _groups_todo(datasets, groups, params, make_estimator, checkpoint_dir, only_groups):
    """Indices of the groups that still have to be COMPUTED: in only_groups (if given) and no valid checkpoint."""
    only = _check_only_groups(groups, only_groups)
    n_datasets = len(datasets)
    todo = []
    for g, group in enumerate(groups):
        if only is not None and group['name'] not in only:
            continue
        test_idx = list(group['dataset_idx'])
        pool_idx = [i for i in range(n_datasets) if i not in test_idx]
        if checkpoint_dir is not None:
            path = Path(checkpoint_dir) / f"group_{g:02d}.npz"
            sig = _group_signature(params, group, datasets, test_idx, pool_idx, make_estimator)
            if path.exists() and _load_checkpoint(path, sig) is not None:
                continue
        todo.append(g)
    return todo


def _make_params(window_size, start_frame, stop_frame, step, n_folds, seed, make_estimator, extra_params=None):
    params = {'window_size': int(window_size), 'start_frame': int(start_frame), 'stop_frame': int(stop_frame),
              'step': int(step), 'n_folds': int(n_folds), 'seed': int(seed),
              'estimator': repr(make_estimator())}
    if extra_params:
        params.update(extra_params)
    return params


def _process_window(datasets, pool_idx, test_idx, start, window_size, folds, y_pool_frames, y_test_frames,
                    make_estimator, n_jobs):
    """
    Everything done for ONE window of ONE left-out group: build the pooled frame samples and the
    left-out test sets, fit and evaluate the n_folds models (threads if n_jobs != 1).
    Shared by run_nested_loso and estimate_runtime, so the benchmark times exactly the real code path.
    Returns the list of _fit_eval_fold outputs, one per fold.
    """
    end = start + window_size
    X_pool = np.concatenate([window_frame_samples(datasets[i]['X'], start, end) for i in pool_idx], axis=0)
    test_sets = [(window_frame_samples(datasets[j]['X'], start, end), y_test_frames[jj])
                 for jj, j in enumerate(test_idx)]
    if n_jobs == 1:
        out = [_fit_eval_fold(make_estimator, X_pool, y_pool_frames, f['train_rows'], f['test_rows'],
                              test_sets, window_size) for f in folds]
    else:
        out = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_fit_eval_fold)(make_estimator, X_pool, y_pool_frames, f['train_rows'], f['test_rows'],
                                    test_sets, window_size) for f in folds)
    return out


def run_nested_loso(datasets, groups, make_estimator, window_size=5, start_frame=1, stop_frame=100,
                    step=1, n_folds=10, seed=0, n_jobs=1, checkpoint_dir=None,
                    extra_params=None, only_groups=None, verbose=True):
    """
    Nested leave-one-face-out grand model (see the module docstring).

    datasets : output of load_all_datasets (z-scored, X: (trials, frames, pixels))
    groups : output of build_leave_out_groups
    make_estimator : callable -> fresh estimator, e.g. lambda: LinearSVC(C=0.0001, max_iter=10000)
    n_jobs : threads used to fit the n_folds models of a window in parallel
    checkpoint_dir : if given, every finished group (= one left-out face, all its windows) is saved there as
        group_XX.npz and skipped on a re-run (only if the parameters, datasets and estimator are identical)
    only_groups : None = compute all groups; or a list of group names, e.g. ['face 2'], to compute only those
        (pilot run). Groups with a valid checkpoint are always loaded; the others stay NaN and
        results['groups_done'] says which groups are filled. A pilot group computed with the real settings
        is saved as a normal checkpoint, so the full run reuses it.

    Returns results dict (arrays):
        'dataset_ids', 'dataset_face_ids', 'dataset_scramble', 'n_trials', 'group_of_dataset' (n_datasets)
        'group_names', 'group_face_ids'                                                (n_groups)
        'centers', 'time_ms'                                                           (n_windows)
        'ho_frame_acc', 'ho_trial_acc'     (n_datasets, n_folds, n_windows) - left-out dataset accuracy
        'inner_frame_acc', 'inner_trial_acc' (n_groups, n_folds, n_windows) - within-pool accuracy
        'w_mean_windows', 'w_std_windows'  (n_groups, n_windows, n_pixels) float32 - mean / SD (ddof=1)
                                           of the n_folds weight vectors of a group
        'w_fold_corr'                      (n_groups, n_windows) - mean pairwise Pearson r between the
                                           n_folds weight maps of a group
        'groups_done'                      (n_groups,) bool - computed or loaded from a checkpoint
        'params'                           dict
    """
    n_datasets = len(datasets)
    n_groups = len(groups)
    n_frames = datasets[0]['X'].shape[1]
    n_pixels = datasets[0]['X'].shape[2]
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if window_size % 2 == 0:
        warnings.warn("window_size is even: ties in the majority vote are assigned to class 0")

    group_of_dataset = np.full(n_datasets, -1, dtype=int)
    for g, group in enumerate(groups):
        for i in group['dataset_idx']:
            if group_of_dataset[i] != -1:
                raise ValueError(f"dataset {datasets[i]['id']} is in more than one group")
            group_of_dataset[i] = g
    if (group_of_dataset < 0).any():
        raise ValueError("some datasets are not in any group")

    starts = window_bounds(n_frames, window_size, start_frame, stop_frame, step)
    centers = np.array([s + window_size // 2 for s in starts])
    n_windows = len(starts)
    time_ms = centers_to_time_ms(centers)

    params = _make_params(window_size, start_frame, stop_frame, step, n_folds, seed, make_estimator, extra_params)

    inner_frame = np.full((n_groups, n_folds, n_windows), np.nan)
    inner_trial = np.full((n_groups, n_folds, n_windows), np.nan)
    ho_frame = np.full((n_datasets, n_folds, n_windows), np.nan)
    ho_trial = np.full((n_datasets, n_folds, n_windows), np.nan)
    w_mean = np.full((n_groups, n_windows, n_pixels), np.nan, dtype=np.float32)
    w_std = np.full((n_groups, n_windows, n_pixels), np.nan, dtype=np.float32)
    w_fold_corr = np.full((n_groups, n_windows), np.nan)
    groups_done = np.zeros(n_groups, dtype=bool)

    only = _check_only_groups(groups, only_groups)
    todo = _groups_todo(datasets, groups, params, make_estimator, checkpoint_dir, only_groups)

    t_run = time.time()
    n_computed = 0
    for g, group in enumerate(groups):
        test_idx = list(group['dataset_idx'])
        pool_idx = [i for i in range(n_datasets) if i not in test_idx]
        signature = _group_signature(params, group, datasets, test_idx, pool_idx, make_estimator)
        ckpt_path = Path(checkpoint_dir) / f"group_{g:02d}.npz" if checkpoint_dir else None

        if ckpt_path is not None and ckpt_path.exists():
            ck = _load_checkpoint(ckpt_path, signature)
            if ck is not None:
                inner_frame[g] = ck['inner_frame']
                inner_trial[g] = ck['inner_trial']
                ho_frame[test_idx] = ck['ho_frame']
                ho_trial[test_idx] = ck['ho_trial']
                w_mean[g] = ck['w_mean']
                w_std[g] = ck['w_std']
                w_fold_corr[g] = ck['w_fold_corr']
                groups_done[g] = True
                if verbose:
                    print(f"[group {g + 1}/{n_groups}] {group['name']}: loaded from checkpoint")
                continue

        if only is not None and group['name'] not in only:
            if verbose:
                print(f"[group {g + 1}/{n_groups}] {group['name']}: skipped (not in only_groups)")
            continue

        t_group = time.time()
        pool_y = np.concatenate([datasets[i]['y'] for i in pool_idx])
        y_pool_frames = np.repeat(pool_y, window_size)
        folds = make_inner_folds(pool_y, n_folds, seed + g, window_size)
        y_test_frames = [np.repeat(datasets[j]['y'], window_size) for j in test_idx]

        if verbose:
            print(f"[group {g + 1}/{n_groups}] left out: {group['name']} "
                  f"({', '.join(datasets[j]['id'] for j in test_idx)}) | pool: {len(pool_idx)} datasets, "
                  f"{len(pool_y)} trials | {n_folds} folds x {n_windows} windows")

        for w, start in enumerate(starts):
            out = _process_window(datasets, pool_idx, test_idx, start, window_size, folds,
                                  y_pool_frames, y_test_frames, make_estimator, n_jobs)

            W = np.empty((n_folds, n_pixels))
            for k, (i_frame, i_trial, ho, wvec) in enumerate(out):
                inner_frame[g, k, w] = i_frame
                inner_trial[g, k, w] = i_trial
                for jj, j in enumerate(test_idx):
                    ho_frame[j, k, w] = ho[jj][0]
                    ho_trial[j, k, w] = ho[jj][1]
                W[k] = wvec
            w_mean[g, w] = W.mean(axis=0)
            w_std[g, w] = W.std(axis=0, ddof=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                c = np.corrcoef(W)
            w_fold_corr[g, w] = float(np.nanmean(c[np.triu_indices(n_folds, 1)]))

            if verbose and ((w + 1) % 10 == 0 or w + 1 == n_windows):
                print(f"    window {w + 1}/{n_windows} ({float(time_ms[w]):.0f} ms) | "
                      f"{time.time() - t_group:.0f} s in this group")

        if ckpt_path is not None:
            _save_checkpoint(ckpt_path, signature,
                             inner_frame=inner_frame[g], inner_trial=inner_trial[g],
                             ho_frame=ho_frame[test_idx], ho_trial=ho_trial[test_idx],
                             w_mean=w_mean[g], w_std=w_std[g], w_fold_corr=w_fold_corr[g])

        groups_done[g] = True
        n_computed += 1
        if verbose:
            elapsed = time.time() - t_run
            remaining = (len(todo) - n_computed) * elapsed / n_computed
            ho_peak = float(np.nanmean(ho_trial[test_idx]))
            print(f"    group done in {time.time() - t_group:.0f} s | mean left-out trial acc over all windows "
                  f"{ho_peak:.3f} | elapsed {elapsed / 60:.1f} min, ~{remaining / 60:.1f} min left")

    results = {
        'dataset_ids': np.array([d['id'] for d in datasets]),
        'dataset_face_ids': np.array([int(d['face_id']) for d in datasets]),
        'dataset_scramble': np.array([str(d['scramble']) for d in datasets]),
        'n_trials': np.array([len(d['y']) for d in datasets]),
        'group_of_dataset': group_of_dataset,
        'group_names': np.array([g['name'] for g in groups]),
        'group_face_ids': np.array([int(g['face_id']) for g in groups]),
        'centers': centers,
        'time_ms': time_ms,
        'ho_frame_acc': ho_frame,
        'ho_trial_acc': ho_trial,
        'inner_frame_acc': inner_frame,
        'inner_trial_acc': inner_trial,
        'w_mean_windows': w_mean,
        'w_std_windows': w_std,
        'w_fold_corr': w_fold_corr,
        'groups_done': groups_done,
        'params': params,
    }
    return results


def _fmt_duration(sec):
    sec = float(sec)
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 5400:
        return f"{sec / 60:.0f} min"
    return f"{int(sec // 3600)} h {int(round((sec % 3600) / 60))} min"


def estimate_runtime(datasets, groups, make_estimator, window_size=5, start_frame=1, stop_frame=100, step=1,
                     n_folds=10, seed=0, n_jobs=1, checkpoint_dir=None, extra_params=None,
                     only_groups=None, n_bench_windows=2, verbose=True):
    """
    Small run-time calculator. Times a few REAL windows (all n_folds models, the same code path and the same
    n_jobs as the real run) on the largest pool and extrapolates to every window of every group that still has
    to be computed (groups with a valid checkpoint are skipped, exactly as in run_nested_loso).
    Takes about n_bench_windows x one window (seconds to a minute or two).

    Returns dict: sec_per_window, n_windows, n_groups_todo, n_fits, total_sec.
    """
    n_datasets = len(datasets)
    n_frames = datasets[0]['X'].shape[1]
    n_pixels = datasets[0]['X'].shape[2]
    starts = window_bounds(n_frames, window_size, start_frame, stop_frame, step)
    n_windows = len(starts)
    params = _make_params(window_size, start_frame, stop_frame, step, n_folds, seed, make_estimator, extra_params)

    todo = _groups_todo(datasets, groups, params, make_estimator, checkpoint_dir, only_groups)

    n_fits = len(todo) * n_folds * n_windows
    result = {'sec_per_window': 0.0, 'n_windows': n_windows, 'n_groups_todo': len(todo),
              'n_fits': n_fits, 'total_sec': 0.0}
    print("=== Run-time estimate ===")
    if not todo:
        print("nothing left to compute (every requested group already has a valid checkpoint)")
        return result

    g_b = min(todo, key=lambda g: len(groups[g]['dataset_idx']))        # largest pool = fewest left-out datasets
    test_idx = list(groups[g_b]['dataset_idx'])
    pool_idx = [i for i in range(n_datasets) if i not in test_idx]
    pool_y = np.concatenate([datasets[i]['y'] for i in pool_idx])
    y_pool_frames = np.repeat(pool_y, window_size)
    folds = make_inner_folds(pool_y, n_folds, seed + g_b, window_size)
    y_test_frames = [np.repeat(datasets[j]['y'], window_size) for j in test_idx]

    bench_pos = np.linspace(0, n_windows - 1, n_bench_windows + 2)[1:-1].round().astype(int)
    times = []
    for pos in bench_pos:
        t0 = time.time()
        _process_window(datasets, pool_idx, test_idx, starts[int(pos)], window_size, folds,
                        y_pool_frames, y_test_frames, make_estimator, n_jobs)
        times.append(time.time() - t0)
    sec_per_window = float(np.mean(times))
    total_sec = sec_per_window * n_windows * len(todo)

    n_conc = n_folds if n_jobs in (None, -1) else min(int(n_jobs), n_folds)
    pool_gb = len(pool_y) * window_size * n_pixels * 8 / 1e9
    data_gb = sum(d['X'].nbytes for d in datasets) / 1e9
    finish = datetime.datetime.now() + datetime.timedelta(seconds=total_sec)

    print(f"benchmark: {len(times)} windows of group '{groups[g_b]['name']}' (pool {len(pool_idx)} datasets, "
          f"{len(pool_y)} trials), {n_folds} fold models each, n_jobs={n_jobs}: "
          + ", ".join(f"{t:.1f} s" for t in times))
    print(f"per window (all {n_folds} fold models): {sec_per_window:.1f} s")
    print(f"still to run: {len(todo)} of {len(groups)} groups"
          f"{'' if only_groups is None else ' (only_groups: ' + ', '.join(only_groups) + ')'} x {n_windows} windows = "
          f"{len(todo) * n_windows:,} windows = {n_fits:,} fits")
    print(f"estimated total: {_fmt_duration(total_sec)} ({total_sec / 3600:.1f} h), "
          f"finishing around {finish.strftime('%a %H:%M')}")
    print(f"memory (rough): data {data_gb:.1f} GB + about {pool_gb * (1 + 3 * n_conc):.1f} GB for the fits "
          f"({n_conc} fits at a time)")
    cores = os.cpu_count()
    if cores is not None and n_conc > cores:
        print(f"WARNING: n_jobs={n_jobs} but only {cores} CPU cores detected - lower N_JOBS")
    print("(about +/-25%: groups that leave out two datasets have a smaller pool and run slightly faster, "
          "and the first benchmark window includes warm-up)")

    result.update({'sec_per_window': sec_per_window, 'total_sec': total_sec})
    return result


# ===========================================================================
# 4. Saving / loading the model results
# ===========================================================================
def save_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in results.items() if k != 'params'}
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, params_json=np.array(json.dumps(results['params'], default=str)), **arrays)
    tmp.replace(path)
    return path


def load_results(path):
    """Open a saved .npz and return the same dict that run_nested_loso returned."""
    with np.load(path, allow_pickle=False) as d:
        results = {k: d[k] for k in d.files if k != 'params_json'}
        results['params'] = json.loads(str(d['params_json']))
    for key in ('dataset_ids', 'dataset_scramble', 'group_names'):
        results[key] = [str(x) for x in results[key]]
    if 'groups_done' not in results:                      # files saved before the pilot option existed
        results['groups_done'] = np.ones(len(results['group_names']), dtype=bool)
    return results


def save_figure(fig, out_dir, name, dpi=150):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path


# ===========================================================================
# 5. Accuracy summaries and statistics
# ===========================================================================
def group_accuracy(results, metric='trial_acc', source='heldout'):
    """
    Accuracy per group and fold: (n_groups, n_folds, n_windows).
    source='heldout' : accuracy on the left-out group = trial-weighted mean over its datasets
    source='inner'   : accuracy on the model's own held-out ~10% of the pooled trials
    metric : 'trial_acc' (majority vote) or 'frame_acc'
    """
    if source == 'inner':
        return results[f'inner_{metric}']
    acc = results[f'ho_{metric}']
    n_tr = np.asarray(results['n_trials'], dtype=float)
    g_of_d = np.asarray(results['group_of_dataset'])
    n_groups = int(g_of_d.max()) + 1
    out = np.zeros((n_groups,) + acc.shape[1:])
    for g in range(n_groups):
        idx = np.where(g_of_d == g)[0]
        out[g] = np.tensordot(n_tr[idx] / n_tr[idx].sum(), acc[idx], axes=(0, 0))
    return out


def group_curves(results, metric='trial_acc', source='heldout'):
    """(n_groups, n_windows): accuracy averaged over the folds of each group (one number per group)."""
    return group_accuracy(results, metric, source).mean(axis=1)


def _bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


def _wilcoxon_p(d):
    """Two-tailed Wilcoxon signed-rank p-value on differences d."""
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return float(wilcoxon(d, alternative='two-sided', method='auto').pvalue)
        except TypeError:  # older scipy without the `method` argument
            return float(wilcoxon(d, alternative='two-sided').pvalue)


def accuracy_stats_vs_chance(curves, time_ms, chance=0.5, alpha=0.05):
    """
    curves : (n_groups, n_windows), one value per left-out group.
    Per-window Wilcoxon signed-rank vs chance; BH-FDR across post-stimulus windows (time_ms >= 0).
    Returns dict of (n_windows,) arrays: mean, sem, p, q (nan pre-stimulus), sig, post.
    """
    n_units, n_windows = curves.shape
    p = np.array([_wilcoxon_p(curves[:, w] - chance) for w in range(n_windows)])
    post = np.asarray(time_ms) >= 0
    q = np.full(n_windows, np.nan)
    q[post] = _bh_fdr(p[post])
    sig = np.zeros(n_windows, dtype=bool)
    sig[post] = q[post] < alpha
    return {'mean': curves.mean(axis=0), 'sem': curves.std(axis=0, ddof=1) / np.sqrt(n_units),
            'p': p, 'q': q, 'sig': sig, 'post': post}


def first_sustained_significance(sig, time_ms, min_run=3):
    """time_ms of the first window starting a run of >= min_run consecutive significant windows, else None."""
    run = 0
    for i, s in enumerate(sig):
        run = run + 1 if s else 0
        if run >= min_run:
            return float(time_ms[i - min_run + 1])
    return None


def find_peak_window(mean_curve, time_ms):
    """Index of the post-stimulus window with the highest value."""
    post_idx = np.where(np.asarray(time_ms) >= 0)[0]
    return int(post_idx[np.argmax(np.asarray(mean_curve)[post_idx])])


def print_summary(results, metric='trial_acc', chance=0.5, alpha=0.05, min_run=3):
    time_ms = results['time_ms']
    ho = group_curves(results, metric, 'heldout')
    inner = group_curves(results, metric, 'inner')
    stats = accuracy_stats_vs_chance(ho, time_ms, chance, alpha)
    peak = find_peak_window(stats['mean'], time_ms)
    onset = first_sustained_significance(stats['sig'], time_ms, min_run)
    n_folds = results['ho_trial_acc'].shape[1]

    print(f"--- left-out groups, {metric} ---")
    print(f"groups: {ho.shape[0]}, windows: {ho.shape[1]}, folds per group: {n_folds}")
    print(f"peak (post-stimulus): {float(time_ms[peak]):.0f} ms, left-out acc = "
          f"{float(stats['mean'][peak]):.3f} +/- {float(stats['sem'][peak]):.3f} SEM "
          f"(n={ho.shape[0]} groups), q = {float(stats['q'][peak]):.4g}")
    print(f"within-pool acc (own held-out 10%) at the same window: {float(inner[:, peak].mean()):.3f}")
    print(f"significant post-stimulus windows (BH-FDR q<{alpha}): "
          f"{int(stats['sig'].sum())} / {int(stats['post'].sum())}")
    if onset is None:
        print(f"no run of >= {min_run} consecutive significant windows")
    else:
        print(f"first run of >= {min_run} consecutive significant windows starts at {onset:.0f} ms")

    per_fold = group_accuracy(results, metric, 'heldout')[:, :, peak]       # (n_groups, n_folds)
    n_ds = np.bincount(np.asarray(results['group_of_dataset']))
    print(f"per left-out group at {float(time_ms[peak]):.0f} ms (mean +/- SD over the {n_folds} fold models):")
    for g, name in enumerate(results['group_names']):
        print(f"   {name:<9} ({int(n_ds[g])} dataset{'s' if n_ds[g] > 1 else ' '}): "
              f"{float(per_fold[g].mean()):.3f} +/- {float(per_fold[g].std(ddof=1)):.3f}")


# ===========================================================================
# 6. Accuracy figures
# ===========================================================================
def plot_heldout_accuracy(results, metric='trial_acc', chance=0.5, alpha=0.05, title=None, ax=None):
    """
    Accuracy on the LEFT-OUT group over time: thin line per group (mean of its fold models),
    thick line = mean +/- SEM across groups, black squares = windows significantly above/below
    chance (Wilcoxon over groups, BH-FDR across post-stimulus windows).
    Returns (fig, stats).
    """
    time_ms = results['time_ms']
    curves = group_curves(results, metric, 'heldout')
    stats = accuracy_stats_vs_chance(curves, time_ms, chance, alpha)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    for g in range(curves.shape[0]):
        ax.plot(time_ms, curves[g], color='gray', lw=0.6, alpha=0.5)
    ax.plot(time_ms, stats['mean'], color='black', lw=2.2, label=f'mean across {curves.shape[0]} left-out groups')
    ax.fill_between(time_ms, stats['mean'] - stats['sem'], stats['mean'] + stats['sem'],
                    color='black', alpha=0.25, label='SEM')
    ax.axhline(chance, color='gray', ls='--', lw=1, label='chance')
    ax.axvline(0, color='red', ls='--', lw=1, label='stimulus onset')
    ax.set_ylim(min(0.4, float(curves.min()) - 0.02), 1.06)
    if stats['sig'].any():
        ax.plot(time_ms[stats['sig']], np.full(int(stats['sig'].sum()), 1.03), 's', color='black', ms=3.5,
                label=f'sig. vs chance (q<{alpha})')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Accuracy on the left-out group')
    ax.set_title(title or f"Nested leave-one-face-out - left-out {metric}")
    ax.legend(loc='lower right', fontsize=8)
    return fig, stats


def plot_inner_vs_heldout(results, metric='trial_acc', chance=0.5):
    """
    Within-pool accuracy (each model on its own held-out ~10% of the mixed pooled trials) vs accuracy
    on the left-out face (never seen in training). Bands = SEM across the groups; the within-pool
    curves of different groups come from heavily overlapping pools, so they are descriptive only.
    """
    time_ms = results['time_ms']
    fig, ax = plt.subplots(figsize=(9, 5))
    for source, color, label in [('inner', 'tab:blue', 'within-pool (own held-out 10%)'),
                                 ('heldout', 'tab:orange', 'left-out face (cross-dataset)')]:
        curves = group_curves(results, metric, source)
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
        ax.plot(time_ms, mean, color=color, lw=2, label=label)
        ax.fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.axhline(chance, color='gray', ls='--', lw=1, label='chance')
    ax.axvline(0, color='red', ls='--', lw=1, label='stimulus onset')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f"Within-pool vs left-out-face accuracy ({metric})")
    ax.legend(loc='lower right', fontsize=8)
    return fig


def plot_generalization_gap(results, metric='trial_acc'):
    """Within-pool minus left-out accuracy per group over time (descriptive): how much is lost on a new face."""
    time_ms = results['time_ms']
    gap = group_curves(results, metric, 'inner') - group_curves(results, metric, 'heldout')
    mean = gap.mean(axis=0)
    sem = gap.std(axis=0, ddof=1) / np.sqrt(gap.shape[0])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for g in range(gap.shape[0]):
        ax.plot(time_ms, gap[g], color='gray', lw=0.6, alpha=0.5)
    ax.plot(time_ms, mean, color='black', lw=2.2, label='mean across groups')
    ax.fill_between(time_ms, mean - sem, mean + sem, color='black', alpha=0.25, label='SEM')
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.axvline(0, color='red', ls='--', lw=1, label='stimulus onset')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('within-pool acc. - left-out acc.')
    ax.set_title(f"Generalization gap to a new face ({metric})")
    ax.legend(loc='upper left', fontsize=8)
    return fig


def plot_frame_vs_trial(results, title="Left-out face: frame-level vs trial-level accuracy"):
    """Mean +/- SEM across groups of frame-level and trial-level (majority vote) accuracy."""
    time_ms = results['time_ms']
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, color, label in [('frame_acc', 'tab:blue', 'frame-level'),
                                 ('trial_acc', 'tab:orange', 'trial-level (majority vote)')]:
        curves = group_curves(results, metric, 'heldout')
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
        ax.plot(time_ms, mean, color=color, lw=2, label=label)
        ax.fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=0.25)
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='chance')
    ax.axvline(0, color='red', ls='--', lw=1, label='stimulus onset')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Accuracy on the left-out group')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=8)
    return fig


def plot_dataset_heatmap(results, metric='trial_acc'):
    """Datasets x windows heatmap of left-out accuracy (mean over the fold models), rows ordered by group."""
    time_ms = results['time_ms']
    acc = results[f'ho_{metric}'].mean(axis=1)                       # (n_datasets, n_windows)
    g_of_d = np.asarray(results['group_of_dataset'])
    order = np.lexsort((np.arange(len(g_of_d)), g_of_d))
    acc = acc[order]
    labels = [f"{results['dataset_ids'][i]} | face {int(results['dataset_face_ids'][i])} | "
              f"{results['dataset_scramble'][i]}" for i in order]
    n = len(order)

    fig, ax = plt.subplots(figsize=(10, 0.38 * n + 2))
    im = ax.imshow(acc, aspect='auto', vmin=0, vmax=1, cmap='viridis',
                   extent=[time_ms[0], time_ms[-1], n, 0])
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels, fontsize=7)
    boundaries = np.where(np.diff(g_of_d[order]) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b, color='white', lw=1.2)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_title(f"Left-out accuracy per dataset ({metric}); white lines separate the left-out groups")
    fig.colorbar(im, ax=ax, label='accuracy')
    return fig


def plot_peak_window_groups(results, metric='trial_acc', chance=0.5):
    """Left-out accuracy of every group at the peak post-stimulus window: mean +/- SD over its fold models."""
    time_ms = results['time_ms']
    mean_curve = group_curves(results, metric, 'heldout').mean(axis=0)
    peak = find_peak_window(mean_curve, time_ms)
    per_fold = group_accuracy(results, metric, 'heldout')[:, :, peak]
    means = per_fold.mean(axis=1)
    sds = per_fold.std(axis=1, ddof=1)
    n_ds = np.bincount(np.asarray(results['group_of_dataset']))
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(np.arange(len(order)), means[order], yerr=sds[order], fmt='o', color='black', capsize=3)
    ax.axhline(chance, color='gray', ls='--', lw=1, label='chance')
    ax.axhline(float(means.mean()), color='tab:orange', lw=1.5, label=f'mean = {float(means.mean()):.3f}')
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{results['group_names'][i]}\n({int(n_ds[i])} ds)" for i in order], fontsize=8)
    ax.set_ylim(min(0.4, float((means - sds).min()) - 0.03), 1.02)
    ax.set_ylabel('Accuracy on the left-out group')
    ax.set_title(f"{metric} per left-out group at the peak window ({float(time_ms[peak]):.0f} ms); "
                 f"bars = SD over the fold models")
    ax.legend(loc='lower right', fontsize=8)
    return fig


def plot_fold_spread(results, metric='trial_acc'):
    """SD over the fold models of the left-out accuracy over time: sensitivity to which ~90% of the trials trained the model."""
    time_ms = results['time_ms']
    sd = group_accuracy(results, metric, 'heldout').std(axis=1, ddof=1)       # (n_groups, n_windows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for g in range(sd.shape[0]):
        ax.plot(time_ms, sd[g], color='gray', lw=0.6, alpha=0.6)
    ax.plot(time_ms, sd.mean(axis=0), color='black', lw=2.2, label='mean across groups')
    ax.axvline(0, color='red', ls='--', lw=1, label='stimulus onset')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('SD over fold models')
    ax.set_title(f"Spread of the left-out accuracy across the fold models ({metric})")
    ax.legend(loc='upper left', fontsize=8)
    return fig


# ===========================================================================
# 6b. Single left-out group check (use on a pilot run of one face, or on any computed group)
# ===========================================================================
def _group_index(results, group_name):
    names = list(results['group_names'])
    if group_name not in names:
        raise ValueError(f"unknown group {group_name!r}; available: {names}")
    g = names.index(group_name)
    if not bool(np.asarray(results['groups_done'])[g]):
        raise ValueError(f"group {group_name!r} was not computed in this results file")
    return g


def print_group_check(results, group_name, chance=0.5):
    """Numbers to look at for one left-out group: pre-stimulus sanity check, peak window, fold stability."""
    g = _group_index(results, group_name)
    time_ms = results['time_ms']
    ds_idx = np.where(np.asarray(results['group_of_dataset']) == g)[0]
    n_trials = int(np.asarray(results['n_trials'])[ds_idx].sum())
    ho_t = group_accuracy(results, 'trial_acc', 'heldout')[g]                  # (n_folds, n_windows)
    ho_f = group_accuracy(results, 'frame_acc', 'heldout')[g]
    in_t = results['inner_trial_acc'][g]
    pre = time_ms < 0
    peak = find_peak_window(ho_t.mean(axis=0), time_ms)
    band = 1.96 * np.sqrt(chance * (1 - chance) / n_trials)

    print(f"=== single-group check: left out {group_name} "
          f"({', '.join(results['dataset_ids'][i] for i in ds_idx)}), {n_trials} trials, "
          f"{ho_t.shape[0]} fold models, {ho_t.shape[1]} windows ===")
    print(f"pre-stimulus left-out trial acc (should be ~{chance}): {float(ho_t[:, pre].mean()):.3f} "
          f"(95% band for one model on {n_trials} trials: {chance - band:.2f} to {chance + band:.2f})")
    print(f"peak (post-stimulus) at {float(time_ms[peak]):.0f} ms:")
    print(f"   left-out trial acc  {float(ho_t[:, peak].mean()):.3f} +/- {float(ho_t[:, peak].std(ddof=1)):.3f} SD over the fold models")
    print(f"   left-out frame acc  {float(ho_f[:, peak].mean()):.3f} +/- {float(ho_f[:, peak].std(ddof=1)):.3f}")
    print(f"   within-pool trial acc (own held-out 10%)  {float(in_t[:, peak].mean()):.3f}")
    print(f"   mean weight-map correlation between the fold models  {float(results['w_fold_corr'][g, peak]):.3f}")
    n_above = int((ho_t.mean(axis=0)[~pre] > chance + band).sum())
    print(f"post-stimulus windows above the chance band: {n_above} / {int((~pre).sum())}")
    if len(ds_idx) > 1:
        for d in ds_idx:
            acc_d = results['ho_trial_acc'][d].mean(axis=0)
            print(f"   {results['dataset_ids'][d]} ({results['dataset_scramble'][d]}): "
                  f"{float(acc_d[peak]):.3f} at the peak window")


def plot_group_check_accuracy(results, group_name, chance=0.5):
    """
    Accuracy picture of ONE left-out group: every fold model (thin lines) and their mean on the left-out
    face, against the within-pool accuracy; frame- vs trial-level; spread over the fold models; and the
    correlation between the weight maps of the fold models.
    """
    g = _group_index(results, group_name)
    time_ms = results['time_ms']
    ds_idx = np.where(np.asarray(results['group_of_dataset']) == g)[0]
    n_trials = int(np.asarray(results['n_trials'])[ds_idx].sum())
    ho_t = group_accuracy(results, 'trial_acc', 'heldout')[g]
    ho_f = group_accuracy(results, 'frame_acc', 'heldout')[g]
    in_t = results['inner_trial_acc'][g]
    band = 1.96 * np.sqrt(chance * (1 - chance) / n_trials)
    n_folds = ho_t.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax = axes[0, 0]
    ax.axhspan(chance - band, chance + band, color='gray', alpha=0.15,
               label=f'chance, 95% band for one model ({n_trials} trials)')
    for k in range(n_folds):
        ax.plot(time_ms, ho_t[k], color='tab:orange', lw=0.6, alpha=0.5)
    ax.plot(time_ms, ho_t.mean(axis=0), color='tab:orange', lw=2.4, label=f'left-out {group_name}: mean of the {n_folds} fold models')
    if len(ds_idx) > 1:
        for d, color in zip(ds_idx, ['tab:green', 'tab:red', 'tab:brown']):
            ax.plot(time_ms, results['ho_trial_acc'][d].mean(axis=0), ls='--', lw=1.2, color=color,
                    label=f"   {results['dataset_ids'][d]} ({results['dataset_scramble'][d]})")
    ax.plot(time_ms, in_t.mean(axis=0), color='tab:blue', lw=2, label='within-pool (own held-out 10%)')
    ax.axhline(chance, color='gray', ls='--', lw=1)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_ylim(min(0.4, float(np.nanmin(ho_t)) - 0.02), 1.03)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Trial-level accuracy')
    ax.set_title('Trial-level accuracy')
    ax.legend(fontsize=7, loc='lower right')

    ax = axes[0, 1]
    for arr, color, label in [(ho_t, 'tab:orange', 'trial-level (majority vote)'), (ho_f, 'tab:purple', 'frame-level')]:
        m, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        ax.plot(time_ms, m, color=color, lw=2, label=label)
        ax.fill_between(time_ms, m - sd, m + sd, color=color, alpha=0.2)
    ax.axhline(chance, color='gray', ls='--', lw=1)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Accuracy (mean +/- SD over fold models)')
    ax.set_title('Frame-level vs trial-level on the left-out face')
    ax.legend(fontsize=8, loc='lower right')

    ax = axes[1, 0]
    ax.plot(time_ms, ho_t.std(axis=0, ddof=1), color='tab:orange', lw=2, label='trial-level')
    ax.plot(time_ms, ho_f.std(axis=0, ddof=1), color='tab:purple', lw=2, label='frame-level')
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('SD over the fold models')
    ax.set_title('How much the accuracy depends on which ~90% of the pooled trials trained the model')
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(time_ms, results['w_fold_corr'][g], color='tab:blue', lw=2)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_ylim(min(0.0, float(np.nanmin(results['w_fold_corr'][g])) - 0.05), 1.0)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('mean Pearson r')
    ax.set_title('Weight-map correlation between the fold models', fontsize=9)

    ids = ', '.join(results['dataset_ids'][i] for i in ds_idx)
    fig.suptitle(f"Single-group check - left out {group_name} ({ids})", fontsize=13)
    fig.tight_layout()
    return fig


def plot_group_check_weights(results, group_name, times_ms=(-100, 50, 100, 200, 300), include_peak=True,
                             clip_percentile=99, cmap='RdBu_r'):
    """
    Top row: mean weight map of the group's fold models at chosen times (+ the peak-accuracy window).
    Bottom row: SD of each weight over the fold models (how much the map changes with the training subset).
    """
    g = _group_index(results, group_name)
    time_ms = results['time_ms']
    ho_t = group_accuracy(results, 'trial_acc', 'heldout')[g]

    idxs, labels = [], []
    for t in times_ms:
        i = int(np.argmin(np.abs(time_ms - t)))
        if i not in idxs:
            idxs.append(i)
            labels.append(f"{float(time_ms[i]):.0f} ms")
    if include_peak:
        p = find_peak_window(ho_t.mean(axis=0), time_ms)
        if p in idxs:
            labels[idxs.index(p)] += " (peak acc.)"
        else:
            idxs.append(p)
            labels.append(f"{float(time_ms[p]):.0f} ms (peak acc.)")
    order = np.argsort(idxs)
    idxs, labels = [idxs[k] for k in order], [labels[k] for k in order]

    mean_maps = [_pix_to_img(results['w_mean_windows'][g, i]) for i in idxs]
    sd_maps = [_pix_to_img(results['w_std_windows'][g, i]) for i in idxs]
    vmax = float(np.percentile(np.abs(np.stack(mean_maps)), clip_percentile))
    smax = float(np.percentile(np.stack(sd_maps), clip_percentile))

    ncols = len(idxs)
    fig, axes = plt.subplots(2, ncols, figsize=(2.9 * ncols + 1.2, 6.4), squeeze=False)
    im1 = im2 = None
    for k in range(ncols):
        im1 = axes[0, k].imshow(mean_maps[k], cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[0, k].set_title(labels[k], fontsize=9)
        im2 = axes[1, k].imshow(sd_maps[k], cmap='magma', vmin=0, vmax=smax)
        axes[0, k].axis('off')
        axes[1, k].axis('off')
    fig.colorbar(im1, ax=axes[0].tolist(), shrink=0.85, label='mean weight')
    fig.colorbar(im2, ax=axes[1].tolist(), shrink=0.85, label='SD over fold models')
    fig.suptitle(f"Weight maps of the models trained with {group_name} left out "
                 f"(top: mean over the fold models, bottom: SD over the fold models)", fontsize=12)
    return fig


# ===========================================================================
# 7. Weight figures
# ===========================================================================
def _grand_mean_weights(results):
    """(n_windows, n_pixels): mean over the left-out groups of the group-mean weight maps."""
    return results['w_mean_windows'].astype(float).mean(axis=0)


def _pix_to_img(w):
    n = w.shape[-1]
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError(f"{n} pixels is not a perfect square")
    return np.asarray(w, dtype=float).reshape(side, side)   # C-order == MATLAB reshape(v,100,100)'


def plot_weight_maps(results, times_ms=(-100, 0, 50, 100, 150, 200, 300, 400), include_peak=True,
                     metric='trial_acc', ncols=4, clip_percentile=99, cmap='RdBu_r'):
    """Grid of the grand-mean weight map (mean over groups of the fold-mean maps) at chosen times, shared symmetric scale."""
    time_ms = results['time_ms']
    M = _grand_mean_weights(results)

    idxs, labels = [], []
    for t in times_ms:
        i = int(np.argmin(np.abs(time_ms - t)))
        if i not in idxs:
            idxs.append(i)
            labels.append(f"{float(time_ms[i]):.0f} ms")
    if include_peak:
        p = find_peak_window(group_curves(results, metric, 'heldout').mean(axis=0), time_ms)
        if p in idxs:
            labels[idxs.index(p)] += " (peak acc.)"
        else:
            idxs.append(p)
            labels.append(f"{float(time_ms[p]):.0f} ms (peak acc.)")
    order = np.argsort(idxs)
    idxs, labels = [idxs[k] for k in order], [labels[k] for k in order]

    maps = [_pix_to_img(M[i]) for i in idxs]
    vmax = float(np.percentile(np.abs(np.stack(maps)), clip_percentile))
    nrows = int(np.ceil(len(maps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.1 * nrows), squeeze=False)
    im = None
    for k, ax in enumerate(axes.ravel()):
        if k < len(maps):
            im = ax.imshow(maps[k], cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(labels[k], fontsize=10)
        ax.axis('off')
    fig.suptitle("Grand-mean SVM weight map (mean over left-out groups of the fold-mean maps)", fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='weight')
    return fig


def plot_group_weight_maps(results, window_ms=None, metric='trial_acc', ncols=4, clip_percentile=99, cmap='RdBu_r'):
    """
    One weight map per left-out group (mean of its fold models) at one window (default: the peak-accuracy
    window), shared symmetric scale: does the decoder look the same whichever face was left out?
    """
    time_ms = results['time_ms']
    if window_ms is None:
        w = find_peak_window(group_curves(results, metric, 'heldout').mean(axis=0), time_ms)
    else:
        w = int(np.argmin(np.abs(time_ms - window_ms)))
    maps = [_pix_to_img(results['w_mean_windows'][g, w]) for g in range(results['w_mean_windows'].shape[0])]
    vmax = float(np.percentile(np.abs(np.stack(maps)), clip_percentile))
    nrows = int(np.ceil(len(maps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.1 * nrows), squeeze=False)
    im = None
    n_ds = np.bincount(np.asarray(results['group_of_dataset']))
    for k, ax in enumerate(axes.ravel()):
        if k < len(maps):
            im = ax.imshow(maps[k], cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(f"left out: {results['group_names'][k]} ({int(n_ds[k])} ds)", fontsize=9)
        ax.axis('off')
    fig.suptitle(f"Weight map of each group's models at {float(time_ms[w]):.0f} ms", fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='weight')
    return fig


def plot_weight_norm_over_time(results):
    """L2 norm of the fold-mean weight vector per window; mean +/- SD across the left-out groups."""
    time_ms = results['time_ms']
    norms = np.linalg.norm(results['w_mean_windows'].astype(float), axis=2)    # (n_groups, n_windows)
    mean, sd = norms.mean(axis=0), norms.std(axis=0, ddof=1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for g in range(norms.shape[0]):
        ax.plot(time_ms, norms[g], color='gray', lw=0.6, alpha=0.6)
    ax.plot(time_ms, mean, color='black', lw=2)
    ax.fill_between(time_ms, mean - sd, mean + sd, color='black', alpha=0.25, label='SD across groups')
    ax.axvline(0, color='red', ls='--', lw=1, label='stimulus onset')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('||w||  (L2 norm)')
    ax.set_title('Weight vector magnitude over time')
    ax.legend(fontsize=8)
    return fig


def _corr_with_reference(W, ref):
    """Pearson r between W[g, t, :] and ref[t, :] -> (n_groups, n_windows)."""
    Wc = W - W.mean(axis=2, keepdims=True)
    rc = ref - ref.mean(axis=1, keepdims=True)
    num = np.einsum('gtp,tp->gt', Wc, rc)
    den = np.linalg.norm(Wc, axis=2) * np.linalg.norm(rc, axis=1)[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def plot_weight_stability(results):
    """
    Top: Pearson r between the weight maps of different left-out groups (mean and minimum over group pairs)
    and the mean r between the fold models of the same group (w_fold_corr).
    Bottom: correlation of each group's map with the grand-mean map (row = left-out group).
    """
    time_ms = results['time_ms']
    W = results['w_mean_windows'].astype(float)
    n_groups, n_windows, _ = W.shape
    iu = np.triu_indices(n_groups, 1)

    pair_mean = np.empty(n_windows)
    pair_min = np.empty(n_windows)
    with np.errstate(invalid='ignore', divide='ignore'):
        for t in range(n_windows):
            C = np.corrcoef(W[:, t, :])
            pair_mean[t] = np.nanmean(C[iu])
            pair_min[t] = np.nanmin(C[iu])
    cm = _corr_with_reference(W, _grand_mean_weights(results))

    fig = plt.figure(figsize=(10, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.03], height_ratios=[1, 1.5], wspace=0.04, hspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])
    ax1.tick_params(labelbottom=False)
    ax1.plot(time_ms, pair_mean, color='black', lw=2, label='between groups: mean pairwise r')
    ax1.plot(time_ms, pair_min, color='tab:red', lw=1.2, ls='--', label='between groups: min pairwise r')
    ax1.plot(time_ms, np.nanmean(results['w_fold_corr'], axis=0), color='tab:blue', lw=2,
             label='within a group: mean r between the fold models')
    ax1.axvline(0, color='red', ls='--', lw=1)
    ax1.set_ylabel('Pearson r between weight maps')
    ax1.set_title('Weight-map stability: between left-out groups and between fold models')
    ax1.legend(fontsize=8, loc='lower right')

    im = ax2.imshow(cm, aspect='auto', cmap='viridis', vmin=float(np.nanmin(cm)), vmax=1.0,
                    extent=[time_ms[0], time_ms[-1], n_groups, 0])
    ax2.set_yticks(np.arange(n_groups) + 0.5)
    ax2.set_yticklabels(results['group_names'], fontsize=8)
    ax2.axvline(0, color='red', ls='--', lw=1)
    ax2.set_ylabel('Left-out group')
    ax2.set_xlabel('Time from stimulus onset (ms)')
    ax2.set_title("Correlation of each group's weight map with the grand-mean map")
    fig.colorbar(im, cax=cax, label='Pearson r')
    return fig


def plot_weight_pattern_similarity(results):
    """Window x window Pearson r of the grand-mean weight maps: does the spatial pattern change over time?"""
    time_ms = results['time_ms']
    M = _grand_mean_weights(results)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.corrcoef(M)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1, extent=[time_ms[0], time_ms[-1], time_ms[-1], time_ms[0]])
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Time from stimulus onset (ms)')
    ax.set_title('Similarity of the grand-mean weight maps across time')
    fig.colorbar(im, ax=ax, label='Pearson r')
    return fig


def plot_pixel_time_heatmap(results, normalize_rows=True, cmap='RdBu_r'):
    """
    Pixels x windows heatmap of the grand-mean weights, pixels sorted by the time of their peak |weight|.
    normalize_rows divides each pixel by its own max |weight| (shows WHEN a pixel matters, not how much).
    """
    time_ms = results['time_ms']
    P = _grand_mean_weights(results).T                                      # (n_pixels, n_windows)
    P = P[np.argsort(np.argmax(np.abs(P), axis=1), kind='stable')]
    if normalize_rows:
        row_max = np.abs(P).max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        P = P / row_max
        vmax, label = 1.0, 'weight / max |weight| of the pixel'
    else:
        vmax, label = float(np.percentile(np.abs(P), 99)), 'weight'
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(P, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                   extent=[time_ms[0], time_ms[-1], P.shape[0], 0], interpolation='nearest')
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Pixels (sorted by time of peak |weight|)')
    ax.set_title('Grand-mean weight of every pixel over time')
    fig.colorbar(im, ax=ax, label=label)
    return fig