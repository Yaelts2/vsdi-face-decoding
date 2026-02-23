import numpy as np
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import save_results as sr
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from scripts.functions_scripts import feature_extraction as fe
from scripts.functions_scripts import sliding_win as sw

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
    Plot TRIAL-level permutation test results (compatible with your load_permutation_run output).
    """
    shuffled = np.asarray(perm_data["shuffled_scores_trials"], dtype=float).ravel()
    real = float(perm_data["real_trial_acc"])

    sh_mean = float(perm_data["shuffled_trial_mean"])
    sh_std  = float(perm_data["shuffled_trial_std"])
    sh_min  = float(perm_data["shuffled_trial_min"])
    sh_max  = float(perm_data["shuffled_trial_max"])

    p = float(perm_data["p_value_trial_two_tailed"])
    pass05 = bool(perm_data["pass_alpha_0p05_trial"])

    if title is None:
        title = "Permutation Test (Trial-Level Accuracy)"

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(shuffled, bins=bins)
    ax.axvline(real)
    ax.axvline(sh_mean, linestyle="--")
    ax.axvline(sh_mean - sh_std, linestyle=":")
    ax.axvline(sh_mean + sh_std, linestyle=":")

    ax.set_title(title)
    ax.set_xlabel("Trial-level accuracy")
    ax.set_ylabel("Count")

    ax.text(
        0.02, 0.98,
        f"real = {real:.4f}\n"
        f"mean±std = {sh_mean:.4f} ± {sh_std:.4f}\n"
        f"range = {sh_min:.4f} / {sh_max:.4f}\n"
        f"p = {p:.4g}{' *' if pass05 else ''}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
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
            print(
                f"  Permutation {p + 1}/{n_perm} "
                f"| mean_frame_acc={mean_frame:.4f} "
                f"| mean_trial_acc={mean_trial:.4f}"
            )

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







def plot_sliding_window_permutation_trial_level(
        perm_data,
        chance=0.5,
        title="Sliding-window permutation test (trial-level)",
        figsize=(7, 4),
        ylim=(0.4, 1.0),
        show_null_curves=False,
        null_alpha=0.08,
        show_real_sem=True,     # ← added
        real_alpha=0.25         # transparency for real SEM
    ):
    """
    Plot trial-level real curve vs permutation null (mean ± SEM).
    Uses only saved values. No calculations performed here.
    """

    centers = np.asarray(perm_data["centers"])
    real = perm_data["real_trial_curve"]
    real_sem = perm_data.get("real_trial_sem")  # ← added
    null_mean = perm_data["null_trial_mean"]
    null_sem = perm_data["null_trial_sem"]

    if real is None:
        raise ValueError("perm_data['real_trial_curve'] is missing.")
    if null_mean is None or null_sem is None:
        raise ValueError("perm_data must include 'null_trial_mean' and 'null_trial_sem'.")

    real = np.asarray(real, dtype=float)
    null_mean = np.asarray(null_mean, dtype=float)
    null_sem = np.asarray(null_sem, dtype=float)

    if real_sem is not None:
        real_sem = np.asarray(real_sem, dtype=float)

    plt.figure(figsize=figsize)

    # Optional: plot all null curves
    if show_null_curves and (perm_data.get("null_trial_acc") is not None):
        null_curves = np.asarray(perm_data["null_trial_acc"], dtype=float)
        for i in range(null_curves.shape[0]):
            plt.plot(centers, null_curves[i], alpha=null_alpha, linewidth=1)

    # Null mean ± SEM
    plt.plot(centers, null_mean, linewidth=2, label="Null mean (permutations)")
    plt.fill_between(
        centers,
        null_mean - null_sem,
        null_mean + null_sem,
        alpha=0.25,
        label="Null ± SEM"
    )

    # Real mean
    plt.plot(centers, real, linewidth=2, label="Real (trial-level)")

    # Real SEM (if available)
    if show_real_sem and (real_sem is not None):
        plt.fill_between(
            centers,
            real - real_sem,
            real + real_sem,
            alpha=real_alpha,
            label="Real ± SEM"
        )

    # Chance
    if chance is not None:
        plt.axhline(float(chance), linestyle="--", linewidth=1, label=f"Chance={chance}")

    plt.title(title)
    plt.xlabel("Frame (window center)")
    plt.ylabel("Accuracy")

    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend()
    plt.tight_layout()
    plt.show()