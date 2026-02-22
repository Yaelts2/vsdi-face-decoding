import numpy as np
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import save_results as sr
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from scripts.functions_scripts import feature_extraction as fe

import numpy as np

def run_permutation_nested_cv(X,y_trial,
                            groups,
                            n_permutations=100,
                            outer_splitter=None,
                            inner_splitter=None,
                            C_grid=None,
                            metric="acc",
                            rule="one_se",
                            tie_break="smaller_C",
                            random_seed=42,
                            verbose=True):
    """
    Permutation test for nested CV where shuffling is done at the TRIAL level (group level).

    Assumptions
    -----------
    - X is frame-level samples (e.g., (n_frames_total, n_features) 
    - groups identifies the trial/group for each frame-level sample (length = n_samples)
    - y_trial provides ONE label per unique trial/group (length = n_unique_groups)
    and must be aligned with np.unique(groups) order (or you should provide it that way).

    Saves both:
    - frame-level permutation score: perm_result["outer_acc_mean"] or ["outer_auc_mean"]
    - trial-level permutation score: perm_result["outer_acc_trial_mean"] (for metric="acc")
    """
    X = np.asarray(X)
    y_trial = np.asarray(y_trial).astype(int)
    groups = np.asarray(groups)
    if metric not in ("acc", "auc"):
        raise ValueError("metric must be 'acc' or 'auc'.")
    rng = np.random.default_rng(random_seed)
    unique_groups = np.unique(groups)

    # Ensure trial labels correspond 1:1 with unique trials
    if y_trial.size != unique_groups.size:
        raise ValueError(f"y_trial length ({y_trial.size}) must equal number of unique trials in groups ({unique_groups.size}). "
            "y_trial must be aligned with np.unique(groups) order.")

    # Keys from your nested-CV function
    key_frame = "outer_acc_mean" if metric == "acc" else "outer_auc_mean"
    key_trial = "outer_acc_trial_mean" if metric == "acc" else None  # no trial-level AUC in your output dict

    shuffled_scores_frames = np.empty(n_permutations, dtype=float)
    shuffled_scores_trials = np.empty(n_permutations, dtype=float)

    progress_every = max(1, n_permutations // 10)

    for i in range(n_permutations):
        # TRIAL-LEVEL SHUFFLE
        perm_trial_labels = y_trial.copy()
        rng.shuffle(perm_trial_labels)

        # Map shuffled trial labels back to frame-level labels via groups
        y_shuf = np.empty(groups.shape[0], dtype=int)
        for g, new_label in zip(unique_groups, perm_trial_labels):
            y_shuf[groups == g] = new_label

        # Sanity check: within each trial, labels must be constant (trial-level shuffle)
        # (This guarantees we did NOT accidentally shuffle per-frame.)
        if __debug__:
            for g in unique_groups[:min(10, unique_groups.size)]:
                vals = y_shuf[groups == g]
                if not np.all(vals == vals[0]):
                    raise RuntimeError("Permutation labels are not constant within a trial. "
                                    "This indicates shuffling happened below trial level.")

        # ----- Run nested CV on permuted labels -----
        perm_result = cv.run_nested_cv_selectC_then_eval(X,y_shuf, groups=groups,
                                                        outer_splitter=outer_splitter,
                                                        inner_splitter=inner_splitter,
                                                        C_grid=C_grid,
                                                        metric=metric,
                                                        rule=rule,
                                                        tie_break=tie_break,
                                                        n_jobs_inner=1,
                                                        verbose=False)

        # Save frame-level score
        shuffled_scores_frames[i] = float(perm_result[key_frame])
        # Save trial-level score (only for accuracy per your outputs)
        if key_trial is None:
            shuffled_scores_trials[i] = np.nan
        else:
            shuffled_scores_trials[i] = float(perm_result[key_trial])

        if verbose and ((i + 1) % progress_every == 0 or (i + 1) == n_permutations):
            msg = (
                f"  Permutation {i + 1}/{n_permutations} "
                f"frame_{metric}={shuffled_scores_frames[i]:.4f}"
            )
            if metric == "acc":
                msg += f" | trial_acc={shuffled_scores_trials[i]:.4f}"
            print(msg)

    return {"shuffled_scores_frames": shuffled_scores_frames,
            "shuffled_scores_trials": shuffled_scores_trials,
            "n_permutations": int(n_permutations),
            "random_seed": int(random_seed),
            "metric": metric,
            "chance_level": 0.5,
            "trial_id_order": unique_groups # for reference, which trial corresponds to which shuffled label
            }



def permutation_significance_test(real_result,perm_result,
                                metric="acc",
                                level="trial",            # "trial" or "frame"
                                chance_level=0.5):
    """
    Permutation significance test using distance-from-chance, at TRIAL or FRAME level.

    Parameters
    ----------
    real_result : dict
        Output of run_nested_cv_selectC_then_eval on REAL labels.
        Must contain:
        - frame acc:  outer_acc_mean
        - trial acc:  outer_acc_trial_mean
        - frame auc:  outer_auc_mean
    perm_result : dict
        Output of run_permutation_nested_cv 
        Must contain:
        - shuffled_scores_frames
        - shuffled_scores_trials
    metric : {"acc","auc"}
    level : {"trial","frame"}
        Which level to compute significance for.
        - "trial": uses outer_acc_trial_mean and shuffled_scores_trials (recommended for your need)
        - "frame": uses outer_acc_mean/outer_auc_mean and shuffled_scores_frames
    chance_level : float

    Returns
    stats : dict
        Summary + two-tailed p-value.
    """
    if metric not in ("acc", "auc"):
        raise ValueError("metric must be 'acc' or 'auc'.")
    if level not in ("trial", "frame"):
        raise ValueError("level must be 'trial' or 'frame'.")

    # --- choose keys based on requested level ---
    if level == "trial":
        if metric != "acc":
            raise ValueError("Trial-level significance is supported for metric='acc' only (no trial-level AUC saved).")

        real_key = "outer_acc_trial_mean"
        perm_key = "shuffled_scores_trials"

    else:  # level == "frame"
        real_key = "outer_acc_mean" if metric == "acc" else "outer_auc_mean"
        perm_key = "shuffled_scores_frames"

    if real_key not in real_result:
        raise KeyError(f"real_result missing key '{real_key}'. Available keys: {list(real_result.keys())}")
    if perm_key not in perm_result:
        raise KeyError(f"perm_result missing key '{perm_key}'. Available keys: {list(perm_result.keys())}")

    real_score = float(real_result[real_key])

    shuffled_scores = np.asarray(perm_result[perm_key], dtype=float)
    if shuffled_scores.ndim != 1 or shuffled_scores.size == 0:
        raise ValueError(f"perm_result['{perm_key}'] must be a non-empty 1D array.")

    # two-tailed: compare distance-from-chance
    p_value = float(np.mean(np.abs(shuffled_scores - chance_level) >= np.abs(real_score - chance_level)))

    return {
        "metric": metric,
        "level": level,
        "chance_level": float(chance_level),
        "real_score": real_score,
        "shuffled_scores": shuffled_scores,
        "shuffled_mean": float(np.mean(shuffled_scores)),
        "shuffled_std": float(np.std(shuffled_scores, ddof=1)) if shuffled_scores.size > 1 else float("nan"),
        "shuffled_min": float(np.min(shuffled_scores)),
        "shuffled_max": float(np.max(shuffled_scores)),
        "p_value_two_tailed": p_value,
        "pass_alpha_0p05": bool(p_value < 0.05),
    }




def plot_permutation_test_trial(perm_data, bins=30, figsize=(7, 4), title=None):
    """
    Plot TRIAL-level permutation test results.

    Required keys in perm_data:
    - shuffled_scores_trials
    - real_trial_acc
    - shuffled_trial_mean
    - shuffled_trial_std
    - shuffled_trial_min
    - shuffled_trial_max
    - p_value_trial_two_tailed
    - pass_alpha_0p05_trial
    """

    shuffled = np.asarray(perm_data["shuffled_scores_trials"], dtype=float)
    real = float(perm_data["real_trial_acc"])

    sh_mean = float(perm_data["shuffled_trial_mean"])
    sh_std  = float(perm_data["shuffled_trial_std"])
    sh_min  = float(perm_data["shuffled_trial_min"])
    sh_max  = float(perm_data["shuffled_trial_max"])

    p = float(perm_data["p_value_trial_two_tailed"])
    pass05 = bool(perm_data["pass_alpha_0p05_trial"])

    if title is None:
        title = "Permutation Test (Trial-Level Accuracy)"

    plt.figure(figsize=figsize)
    plt.hist(shuffled, bins=bins)
    plt.axvline(real)
    plt.axvline(sh_mean, linestyle="--")
    plt.axvline(sh_mean - sh_std, linestyle=":")
    plt.axvline(sh_mean + sh_std, linestyle=":")

    plt.title(title)
    plt.xlabel("Trial-level accuracy")
    plt.ylabel("Count")

    plt.text(
        0.02, 0.98,
        f"real = {real:.4f}\n"
        f"mean±std = {sh_mean:.4f} ± {sh_std:.4f}\n"
        f"range = {sh_min:.4f} / {sh_max:.4f}\n"
        f"p = {p:.4g}  {'*' if pass05 else ''}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout()
    return plt.gca()







######### sliding window

def sliding_window_permutation_test(X_pix_frames_trials,   # (pixels, frames, trials)
                                    y_trials,              # (trials,)
                                    make_estimator,        # callable -> fresh estimator each fold
                                    window_size=5,
                                    start_frame=15,
                                    stop_frame=125,
                                    step=1,
                                    n_splits=5,
                                    n_perm=100,
                                    seed=0,
                                    return_null=False):
    """
    sliding-window permutation test.

    For each permutation:
        - Shuffle trial labels ONCE
        - Run full sliding-window decoding
        - Store entire accuracy curve

    Returns:
        - null mean ± SEM across permutations (per window)
        - optionally full null curves (n_perm, n_windows)
    """

    X = np.asarray(X_pix_frames_trials)
    y_trials = np.asarray(y_trials).astype(int)

    n_pixels, n_frames, n_trials = X.shape
    if y_trials.shape[0] != n_trials:
        raise ValueError(f"y_trials has {y_trials.shape[0]} but X has {n_trials} trials")

    stop_frame = min(int(stop_frame), n_frames)
    last_start = min(n_frames - window_size, stop_frame - window_size)
    if last_start < start_frame:
        raise ValueError("stop_frame too early for window_size")

    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=n_splits)

    # Precompute window centers
    centers = []
    window_starts = []
    for start in range(start_frame, last_start + 1, step):
        centers.append(start + window_size // 2)
        window_starts.append(start)

    centers = np.asarray(centers)
    n_windows = len(window_starts)

    # Storage for full null curves
    null_frame_curves = np.zeros((n_perm, n_windows)) # (n_perm, n_windows)
    null_trial_curves = np.zeros((n_perm, n_windows)) # (n_perm, n_windows) # stores trial-level accuracy across windows for each permutation

    #               PERMUTATION LOOP
    for p in range(n_perm):
        # Shuffle trial labels ONCE
        y_perm_trials = rng.permutation(y_trials)

        frame_curve = [] # store frame-level accuracy for this permutation across windows
        trial_curve = [] # store trial-level accuracy for this permutation across windows

        #Full sliding window for this permutation
        for w, start in enumerate(window_starts):

            end = start + window_size

            X_win = X[:, start:end, :]
            X_frames, y_perm_frames, groups = fe.frames_as_samples(X_win,y_perm_trials,
                                                                trial_axis=2,
                                                                frame_axis=1,
                                                                pixel_axis=0)

            fold_acc_f = [] # store frame-level accuracy for each fold in this window
            fold_acc_t = [] # store trial-level accuracy for each fold in this window

            for tr_idx, te_idx in gkf.split(X_frames, y_perm_frames, groups):
                clf = make_estimator()
                clf.fit(X_frames[tr_idx], y_perm_frames[tr_idx])
                y_pred = clf.predict(X_frames[te_idx])

                # Frame-level
                acc_f = accuracy_score(y_perm_frames[te_idx], y_pred)
                fold_acc_f.append(float(acc_f))

                # Trial-level
                y_true_trial, y_pred_trial = cv.majority_vote_trial_predictions(y_pred,
                                                                                y_perm_frames[te_idx],
                                                                                groups[te_idx])
                acc_t = accuracy_score(y_true_trial, y_pred_trial)
                fold_acc_t.append(float(acc_t))

            frame_curve.append(np.mean(fold_acc_f))
            trial_curve.append(np.mean(fold_acc_t))

        null_frame_curves[p, :] = frame_curve
        null_trial_curves[p, :] = trial_curve

    #                SUMMARY STATISTICS
    null_frame_mean = np.mean(null_frame_curves, axis=0) # (n_windows,)
    null_frame_sem  = sr._sem(null_frame_curves)  # assumes your SEM works row-wise # (n_windows,)
    null_trial_mean = np.mean(null_trial_curves, axis=0) # (n_windows,)
    null_trial_sem  = sr._sem(null_trial_curves) # (n_windows,)

    out = {"centers": centers,
        "null_frame_acc_mean": null_frame_mean,
        "null_frame_acc_sem":  null_frame_sem,
        "null_trial_acc_mean": null_trial_mean,
        "null_trial_acc_sem":  null_trial_sem,
        "params": {"window_size": window_size,
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "step": step,
                "n_splits": n_splits,
                "n_perm": n_perm,
                "seed": seed,}}

    if return_null:
        out["null_frame_acc"] = null_frame_curves      # (n_perm, n_windows)
        out["null_trial_acc"] = null_trial_curves      # (n_perm, n_windows)
    return out


import numpy as np
import matplotlib.pyplot as plt


def plot_slidingwindow_perm_results(centers,real_curve,null_trial_acc=None,
                                    null_mean=None,
                                    null_sem=None,
                                    chance=0.5,
                                    alpha=0.05,
                                    two_sided=False,
                                    show_pvalues=True,
                                    show_peak_hist=True,
                                    figsize_curve=(8, 4),
                                    figsize_p=(8, 2.8),
                                    figsize_hist=(6, 3.5)):
    """
    Plot sliding-window permutation-test results.

    Panels (controlled by flags):
    1) Accuracy vs time: real curve + null mean ± SEM (or computed from null_trial_acc).
    2) Empirical p-value vs time (optional).
    3) Null histogram at peak time (optional).

    Parameters
    centers : array, shape (n_windows,)
        Window centers (typically frame indices; can be ms if you already converted).
    real_curve : array, shape (n_windows,)
        Real (unshuffled) trial-accuracy per window.
    null_trial_acc : array, shape (n_perm, n_windows), optional
        Full null curves. If provided, p-values + null mean/sem can be computed from it.
    null_mean, null_sem : arrays, shape (n_windows,), optional
        Precomputed null mean/SEM. Used if null_trial_acc not provided.
    chance : float
        Chance level (0.5 for binary classification).
    alpha : float
        Significance threshold to display on p-value panel.
    two_sided : bool
        If True, compute two-sided empirical p-values (based on |null - mean_null| >= |real - mean_null|).
        Otherwise one-sided: P(null >= real).
    show_pvalues : bool
        Whether to draw the p-value panel (requires null_trial_acc).
    show_peak_hist : bool
        Whether to draw histogram of null accuracies at peak real accuracy (requires null_trial_acc).

    Returns
    out : dict
        Computed arrays:
        - centers, real_curve
        - null_mean, null_sem
        - pvals (if computed)
        - peak_idx (if histogram computed)
    """
    centers = np.asarray(centers).ravel()
    real_curve = np.asarray(real_curve, dtype=float).ravel()

    if centers.size != real_curve.size:
        raise ValueError(f"centers and real_curve must match, got {centers.size} vs {real_curve.size}")

    # derive null mean/sem
    pvals = None
    if null_trial_acc is not None:
        null_trial_acc = np.asarray(null_trial_acc, dtype=float)
        if null_trial_acc.ndim != 2 or null_trial_acc.shape[1] != centers.size:
            raise ValueError(
                f"null_trial_acc must be (n_perm, n_windows={centers.size}), got {null_trial_acc.shape}"
            )
        mean_null = np.nanmean(null_trial_acc, axis=0)
        sem_null = np.nanstd(null_trial_acc, axis=0, ddof=1) / np.sqrt(null_trial_acc.shape[0])

        # empirical p-values (with +1 smoothing)
        n_perm = null_trial_acc.shape[0]
        if two_sided:
            dev_real = np.abs(real_curve - mean_null)
            dev_null = np.abs(null_trial_acc - mean_null[None, :])
            pvals = (np.sum(dev_null >= dev_real[None, :], axis=0) + 1) / (n_perm + 1)
        else:
            pvals = (np.sum(null_trial_acc >= real_curve[None, :], axis=0) + 1) / (n_perm + 1)
    else:
        if null_mean is None or null_sem is None:
            raise ValueError("Provide either null_trial_acc OR (null_mean and null_sem).")
        mean_null = np.asarray(null_mean, dtype=float).ravel()
        sem_null = np.asarray(null_sem, dtype=float).ravel()
        if mean_null.size != centers.size or sem_null.size != centers.size:
            raise ValueError("null_mean/null_sem must match centers length.")

    # Panel 1: accuracy curves
    plt.figure(figsize=figsize_curve)
    plt.plot(centers, real_curve, linewidth=2, label="Real accuracy")
    plt.plot(centers, mean_null, linestyle="--", linewidth=1.5, label="Null mean")
    plt.fill_between(centers, mean_null - sem_null, mean_null + sem_null, alpha=0.25, label="Null ± SEM")
    plt.axhline(chance, linestyle=":", linewidth=1.5, label=f"Chance={chance:.2f}")

    # optional significance shading (only if pvals computed)
    if pvals is not None:
        sig = pvals < alpha
        if np.any(sig):
            y0, y1 = plt.ylim()
            plt.fill_between(centers, y0, y1, where=sig, alpha=0.10, step=None, label=f"p < {alpha:g}")

    plt.xlabel("Window center (frame or ms)")
    plt.ylabel("Accuracy")
    plt.title("Sliding-window permutation test")
    plt.legend(frameon=False)
    plt.tight_layout()

    # Panel 2: p-values
    if show_pvalues and (pvals is not None):
        plt.figure(figsize=figsize_p)
        plt.plot(centers, pvals, linewidth=1.8)
        plt.axhline(alpha, linestyle="--", linewidth=1.5)
        plt.ylim(0, 1.0)
        plt.xlabel("Window center (frame or ms)")
        plt.ylabel("Empirical p-value")
        plt.title("Permutation p-values across time")
        plt.tight_layout()

    # Panel 3: histogram at peak
    peak_idx = None
    if show_peak_hist and (null_trial_acc is not None):
        peak_idx = int(np.nanargmax(real_curve))
        null_at_peak = null_trial_acc[:, peak_idx]

        plt.figure(figsize=figsize_hist)
        plt.hist(null_at_peak, bins=20, alpha=0.8)
        plt.axvline(real_curve[peak_idx], linewidth=2, label="Real (peak)")
        plt.axvline(np.nanmean(null_at_peak), linestyle="--", linewidth=1.5, label="Null mean")
        plt.xlabel("Accuracy")
        plt.ylabel("Count")
        plt.title(f"Null distribution at peak (center={centers[peak_idx]})")
        plt.legend(frameon=False)
        plt.tight_layout()

    return {
        "centers": centers,
        "real_curve": real_curve,
        "null_mean": mean_null,
        "null_sem": sem_null,
        "pvals": pvals,
        "peak_idx": peak_idx,
    }