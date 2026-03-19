## function for model control (permutation test, sliding window permutation, etc.)

import numpy as np
from scipy.stats import mannwhitneyu
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import save_results as sr
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from scripts.functions_scripts import feature_extraction as fe
from scripts.functions_scripts import sliding_win as sw
import warnings
import numpy as np
from statsmodels.stats.multitest import multipletests


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



def permutation_significance_test_fixed(real_score, shuffled_scores, chance_level=0.5, one_tailed=True):
    shuffled_scores = np.asarray(shuffled_scores)
    n_shuffle = len(shuffled_scores)
    
    if not one_tailed:
        # Two-tailed: extreme distance in either direction [cite: 20]
        real_dist = np.abs(real_score - chance_level)
        shuffled_dist = np.abs(shuffled_scores - chance_level)
        hits = np.sum(shuffled_dist >= real_dist)
    else:
        # One-tailed: strictly "is real better than null?" [cite: 56]
        hits = np.sum(shuffled_scores >= real_score)

    # Apply the (hits + 1) / (N + 1) formula from your PPT 
    p_value = (hits + 1.0) / (n_shuffle + 1.0)
    
    return {
        "p_value": p_value,
        "pass": p_value < 0.05,
        "shuffled_mean": np.mean(shuffled_scores),
        "shuffled_std": np.std(shuffled_scores, ddof=1)
    }





def plot_permutation_test_trial(perm_data, bins=30, figsize=(7, 4), title=None):
    """
    Fixed plotting function to display p-values correctly using the new logic.
    """
    shuffled = np.asarray(perm_data["shuffled_scores_trials"], dtype=float).ravel()
    real = float(perm_data["real_trial_acc"])
    n_perm = len(shuffled)

    sh_mean = float(perm_data["shuffled_trial_mean"])
    sh_std  = float(perm_data["shuffled_trial_std"])
    sh_min  = float(perm_data["shuffled_trial_min"])
    sh_max  = float(perm_data["shuffled_trial_max"])

    # Update: Use the generic 'p_value_trial' key from your new stats dict
    p = float(perm_data.get("p_value_trial_two_tailed", perm_data.get("p_value_trial", 1.0)))
    pass05 = p < 0.05

    # Logic for "p <" display
    # The smallest possible p-value with (hits+1)/(N+1) is 1/(N+1)
    min_p = 1.0 / (n_perm + 1.0)
    p_text = f"p < {min_p:.4g}" if p <= min_p else f"p = {p:.4f}"

    if title is None:
        title = "Permutation Test (Trial-Level Accuracy)"

    fig, ax = plt.subplots(figsize=figsize)
    
    # Visual improvements
    ax.hist(shuffled, bins=bins, color='lightgray', edgecolor='white', label='Null Distribution')
    ax.axvline(real, color='red', linewidth=2, label='Real Result')
    ax.axvline(sh_mean, color='blue', linestyle="--", label='Null Mean')
    ax.axvline(sh_mean - sh_std, color='blue', linestyle=":", alpha=0.5)
    ax.axvline(sh_mean + sh_std, color='blue', linestyle=":", alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("Trial-level accuracy")
    ax.set_ylabel("Count")
    ax.legend(loc='upper right', fontsize='small')

    # The text box showing the requested 'p <' or 'p =' format
    ax.text(
        0.02, 0.95,
        f"Real result: {real:.4f}\n"
        f"Null mean±std: {sh_mean:.4f} ± {sh_std:.4f}\n"
        f"Null range: [{sh_min:.4f}, {sh_max:.4f}]\n"
        f"{p_text}{' *' if pass05 else ''}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor='white', alpha=0.7),
    )

    fig.tight_layout()
    return ax





######### sliding window


def sliding_window_permutation_test(X_pix_frames_trials,   # (pixels, frames, trials)
                                    y_trials,              # (trials,)
                                    make_estimator,        
                                    window_size=5,
                                    start_frame=15,
                                    stop_frame=125,
                                    step=1,
                                    n_splits=5,
                                    n_perm=100,
                                    seed=0,
                                    return_null=False,
                                    return_perm_stats=False,
                                    verbose=True,   
                                    ):
    """
    Sliding-window permutation test that reuses the REAL decoding pipeline.
    """

    X = np.asarray(X_pix_frames_trials)
    y_trials = np.asarray(y_trials).astype(int)

    if X.ndim != 3:
        raise ValueError(f"X must be 3D (pixels, frames, trials); got shape {X.shape}")

    n_pixels, n_frames, n_trials = X.shape
    if y_trials.shape[0] != n_trials:
        raise ValueError(f"y_trials has {y_trials.shape[0]} but X has {n_trials} trials")

    stop_frame = min(int(stop_frame), n_frames)
    last_start = min(n_frames - window_size, stop_frame - window_size)
    if last_start < start_frame:
        raise ValueError("stop_frame is too early for the given start_frame/window_size")

    centers = np.asarray([s + window_size // 2 for s in range(start_frame, last_start + 1, step)])
    n_windows = centers.size

    rng = np.random.default_rng(seed)

    null_frame_curves = np.zeros((n_perm, n_windows), dtype=float)
    null_trial_curves = np.zeros((n_perm, n_windows), dtype=float)
    null_trial_folds = np.zeros((n_perm, n_windows, n_splits), dtype=float)
    
    perm_stats = [] if return_perm_stats else None

    #progress setup (10%)
    progress_every = max(1, n_perm // 10)

    for p in range(n_perm):
        y_perm = rng.permutation(y_trials)
        perm_out = sw.sliding_window_decode_with_stats(X_pix_frames_trials=X,
                                                    y_trials=y_perm,
                                                    make_estimator=make_estimator,
                                                    window_size=window_size,
                                                    start_frame=start_frame,
                                                    stop_frame=stop_frame,
                                                    step=step,
                                                    n_splits=n_splits)

        if perm_out["centers"].shape[0] != n_windows or not np.all(perm_out["centers"] == centers):
            raise RuntimeError("Permutation decode returned different centers than expected.")

        null_frame_curves[p, :] = np.asarray(perm_out["frame_acc_mean"], dtype=float)
        null_trial_curves[p, :] = np.asarray(perm_out["trial_acc_mean"], dtype=float)
        null_trial_folds[p, :, :] = np.asarray(perm_out["fold_trial_acc"], dtype=float)
        
        
        if return_perm_stats:
            perm_stats.append({
                "frame_acc_mean": perm_out["frame_acc_mean"],
                "frame_acc_std":  perm_out["frame_acc_std"],
                "trial_acc_mean": perm_out["trial_acc_mean"], 
                "trial_acc_std":  perm_out["trial_acc_std"]
            })

        # -------- Progress printing (minimal logic) --------
        if verbose and ((p + 1) % progress_every == 0 or (p + 1) == n_perm):
            mean_frame = float(np.mean(null_frame_curves[p]))
            mean_trial = float(np.mean(null_trial_curves[p]))
            print(f"  Permutation {p + 1}/{n_perm} "
                f"| mean_frame_acc={mean_frame:.4f} "
                f"| mean_trial_acc={mean_trial:.4f}")

    # Summary across permutations
    null_frame_mean = np.mean(null_frame_curves, axis=0)
    null_frame_std  = np.std(null_frame_curves, axis=0)
    null_trial_mean = np.mean(null_trial_curves, axis=0)
    null_trial_std  = np.std(null_trial_curves, axis=0)

    out = {"centers": centers,
        "null_frame_acc_mean": null_frame_mean,
        "null_frame_acc_std":  null_frame_std,
        "null_trial_acc_mean": null_trial_mean,
        "null_trial_acc_std":  null_trial_std,
        "params": {"window_size": int(window_size),
                "start_frame": int(start_frame),
                "stop_frame": int(stop_frame),
                "step": int(step),
                "n_splits": int(n_splits),
                "n_perm": int(n_perm),
                "seed": int(seed)}
    }

    if return_null:
        out["null_frame_acc"] = null_frame_curves          # (n_perm, n_windows)
        out["null_trial_acc"] = null_trial_curves          # (n_perm, n_windows)
        out["null_trial_folds"] = null_trial_folds         # (n_perm, n_windows, n_splits)

    if return_perm_stats:
        out["perm_stats"] = perm_stats

    return out






def sig_vector_perm_mean_across_folds(
    real_fold_trial_acc: np.ndarray,   # (n_window, n_folds)
    null_trial_folds: np.ndarray,      # (n_shuffle, n_window, n_folds)
    alpha: float = 0.001,
    two_sided: bool = True,
    chance_level: float = 0.5,
    fdr_correction: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    sig01     : (n_window,) int   — 1 where significant
    p_raw     : (n_window,) float — uncorrected p-values
    p_correct : (n_window,) float — FDR-corrected p-values (or same as p_raw if fdr_correction=False)
    """
    real = np.asarray(real_fold_trial_acc, dtype=float)
    null = np.asarray(null_trial_folds, dtype=float)

    if real.ndim != 2:
        raise ValueError(f"real must be (n_window, n_folds), got {real.shape}")
    if null.ndim != 3:
        raise ValueError(f"null must be (n_shuffle, n_window, n_folds), got {null.shape}")

    n_window, n_folds = real.shape
    n_shuffle, n_window2, n_folds2 = null.shape

    if n_window2 != n_window or n_folds2 != n_folds:
        raise ValueError(f"Shape mismatch: real={real.shape}, null={null.shape}")

    if np.any(np.isnan(real)) or np.any(np.isnan(null)):
        warnings.warn("NaNs detected — nanmean is used, affected windows may be unreliable.")

    # compute statistics
    real_stat = np.nanmean(real, axis=1)       # (n_window,)
    null_stat = np.nanmean(null, axis=2)       # (n_shuffle, n_window)

    if two_sided:
        real_eff = np.abs(real_stat - chance_level)
        null_eff = np.abs(null_stat - chance_level)
        ge = (null_eff >= real_eff[None, :]).sum(axis=0)
    else:
        ge = (null_stat >= real_stat[None, :]).sum(axis=0)

    p_raw = (ge + 1.0) / (n_shuffle + 1.0)    # (n_window,)

    if fdr_correction:
        _, p_correct, _, _ = multipletests(p_raw, alpha=alpha, method='fdr_bh')
    else:
        p_correct = p_raw

    sig01 = (p_correct < alpha).astype(int)

    return sig01, p_raw, p_correct



def sig_vector_foldlevel_ranksum(fold_trial_acc: np.ndarray,      # (n_window, n_folds)
                                null_trial_folds: np.ndarray,    # (n_shuffle, n_window, n_folds)
                                alpha: float = 0.05,
                                alternative: str = "greater",    # "two-sided", "greater", "less"
                                fdr_correction: bool = True
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    real = np.asarray(fold_trial_acc, dtype=float)
    null = np.asarray(null_trial_folds, dtype=float)

    if real.ndim != 2:
        raise ValueError(f"fold_trial_acc must be (n_window, n_folds). Got {real.shape}")
    if null.ndim != 3:
        raise ValueError(f"null_trial_folds must be (n_shuffle, n_window, n_folds). Got {null.shape}")

    n_window, n_folds = real.shape
    n_shuffle, n_window2, n_folds2 = null.shape

    if n_window2 != n_window or n_folds2 != n_folds:
        raise ValueError(f"Shape mismatch: real={real.shape}, null={null.shape}")

    pvals = np.ones(n_window, dtype=float)

    for w in range(n_window):
        x = real[w, :]
        y = null[:, w, :].ravel()

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if x.size == 0 or y.size == 0:
            continue

        res = mannwhitneyu(x, y, alternative=alternative)
        pvals[w] = res.pvalue

    if fdr_correction:
        _, p_correct, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    else:
        p_correct = pvals

    sig01 = (p_correct < alpha).astype(int)

    return sig01, pvals, p_correct





def plot_sw_perm_simple(prem_results,
                        frames,
                        sig01=None,
                        frame0=27,
                        ms_per_frame=10.0,
                        chance=0.5,
                        title="Sliding-window permutation (trial + frame)",
                        figsize=(7, 4),
                        ylim=(0.4, 1.0),
                        sig_mode="bar",
                        sig_height=0.02):
    """
    Minimal plotting for sliding-window permutation results.

    Expects in prem_results:
    - real_fold_trial_acc : (n_folds, n_windows) OR (n_windows,)
    - real_frame_curve    : (n_windows,)
    - null_trial_folds    : (n_perm, n_windows) OR (n_perm, n_folds, n_windows)

    Parameters
    frames : array-like, (n_windows,)
        Window-center frame indices (same length as curves).
    sig01 : array-like bool/int, (n_windows,), optional
        1/0 vector marking significant windows.
    frame0 : int
        Frame index that will be treated as time 0.
    ms_per_frame : float
        Convert frames to ms: time_ms = (frame - frame0) * ms_per_frame.
        If you prefer "frames relative to 27", set ms_per_frame=1 and rename label as needed.
    """

    # --- extract ---
    real_trial_acc = np.asarray(prem_results["real_trial_curve"], dtype=float)
    real_trial_acc=real_trial_acc[0:45]  

    real_frame_acc = np.asarray(prem_results["real_frame_curve"], dtype=float).ravel()
    real_frame_acc=real_frame_acc[0:45]    
    null_trial_folds = np.asarray(prem_results["null_trial_folds"], dtype=float)
    null_trial_folds=null_trial_folds[:,0:45,:]

    frames = np.asarray(frames, dtype=float).ravel()

    # real trial: mean across folds
    if real_trial_acc.ndim == 2:
        real_trial = real_trial_acc.mean(axis=0)
    else:
        real_trial = real_trial_acc.ravel()

    # null: if (perm, folds, win) -> average folds inside each perm
    if null_trial_folds.ndim == 3:
        null_perm_curves = null_trial_folds.mean(axis=2)         # (n_perm, n_win)
    elif null_trial_folds.ndim == 2:
        null_perm_curves = null_trial_folds                      # (n_perm, n_win)
    else:
        raise ValueError("null_trial_folds must be 2D or 3D.")

    null_mean = null_perm_curves.mean(axis=0)
    null_std  = null_perm_curves.std(axis=0, ddof=1)

    n = real_trial.size
    if frames.size != n or real_frame_acc.size != n or null_mean.size != n:
        raise ValueError("frames/real_trial/real_frame/null must all have the same length.")
    if sig01 is not None:
        sig01 = np.asarray(sig01).astype(bool).ravel()
        if sig01.size != n:
            raise ValueError("sig01 must have the same length as curves.")

    #time axis
    time = (frames - float(frame0)) * float(ms_per_frame)

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time, null_mean, linewidth=2, label="Null mean (trial)")
    ax.fill_between(time, null_mean - 3*null_std, null_mean + 3*null_std, alpha=0.25, label="Null ± Std")

    ax.plot(time, real_trial, linewidth=2.5, label="Real (trial)",color="blue")
    ax.plot(time, real_frame_acc, linestyle="--", linewidth=2, label="Real (frame)",color="purple")

    if chance is not None:
        ax.axhline(float(chance), linestyle="--", linewidth=1, label=f"Chance={chance}")

    if ylim is not None:
        ax.set_ylim(*ylim)

    # significance overlay
    if sig01 is not None and sig01.any():
        y0, y1 = ax.get_ylim()
        yr = (y1 - y0) if (y1 > y0) else 1.0

        if sig_mode == "bar":
            bar_bottom = y0 + 0.02 * yr
            bar_top    = bar_bottom + float(sig_height) * yr
            ax.fill_between(time, bar_bottom, bar_top, where=sig01, step="mid", alpha=0.9)
        elif sig_mode == "dots":
            y_dot = y0 + (0.02 + float(sig_height)) * yr
            ax.scatter(time[sig01], np.full(sig01.sum(), y_dot), s=18)
        else:
            raise ValueError("sig_mode must be 'bar' or 'dots'.")

    ax.set_title(title)
    ax.set_xlabel(f"Time (ms), 0 @ frame {frame0}" if ms_per_frame != 1 else f"Frames (0 @ frame {frame0})")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax