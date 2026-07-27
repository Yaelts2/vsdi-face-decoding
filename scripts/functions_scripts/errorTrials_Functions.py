"""
errorTrials_Functions.py

All functions for Aim 3, Part 2 -- error trial analysis:
1. Data loading & condition selection
2. Normalization (matching correct-trial training stats)
3. Decoding (final model + fold-averaged)
4. Session-level pipeline wrappers (training + error-trial processing)
5. Statistical comparisons (correct vs. error)
6. Plotting
7. Multi-session grand average
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

from scripts.functions_scripts import preprocessing_functions as pre
from scripts.functions_scripts import ml_cv as cv
from scripts.functions_scripts import sliding_win as sw
from scripts.functions_scripts import feature_extraction as fe
from scripts.functions_scripts.save_results import save_experiment, load_data_from_config


# =====================================================================
# 1. DATA LOADING & CONDITION SELECTION
# =====================================================================

def load_error_trials(mat_path):
    """
    Load error-trial VSD data + metadata saved from MATLAB.

    Returns:
        vsd_matrix : (pixels, frames, n_trials) array
        meta : dict with keys cortexTrialId, condId, errType
            (physicalLabel/behavioralLabel are added later by
            select_conditions, once we know which conditions this
            particular model was trained on)
    """
    mat_path = Path(mat_path)
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    vsd_matrix = np.asarray(d["vsdMatrix"])
    s = d["errorTrialsStruct"]

    meta = {"cortexTrialId": np.atleast_1d(s.cortexTrialId).astype(int),
            "condId": np.atleast_1d(s.condId).astype(int),
            "errType": list(np.atleast_1d(s.errType)) }

    n_trials_vsd = vsd_matrix.shape[2]
    n_trials_meta = meta["cortexTrialId"].shape[0]
    if n_trials_vsd != n_trials_meta:
        raise ValueError(f"Mismatch: vsdMatrix has {n_trials_vsd} trials, "
                        f"metadata has {n_trials_meta} trials.")

    print(f"Loaded {n_trials_vsd} error trials (all conditions).")
    return vsd_matrix, meta


def select_conditions(vsd_matrix, meta, face_cond, nonface_cond):
    """
    Filter error trials down to only the face/nonface conditions this
    specific trained model used, and compute physicalLabel/behavioralLabel.

    Parameters
        face_cond, nonface_cond : int, the condId values the model
            was trained on (e.g. 1 and 5)
    """
    cond_id = meta["condId"]
    keep = np.isin(cond_id, [face_cond, nonface_cond])

    vsd_filtered = vsd_matrix[:, :, keep]
    meta_filtered = {k: (np.asarray(v)[keep] if k != "errType" else [v[i] for i in np.where(keep)[0]])
                    for k, v in meta.items()}

    meta_filtered["physicalLabel"] = (meta_filtered["condId"] == face_cond).astype(int)
    meta_filtered["behavioralLabel"] = 1 - meta_filtered["physicalLabel"]

    print(f"Selected {keep.sum()} trials from conditions {face_cond} (face) / {nonface_cond} (nonface).")
    print(f"  face={np.sum(meta_filtered['physicalLabel']==1)}, "
        f"nonface={np.sum(meta_filtered['physicalLabel']==0)}")

    return vsd_filtered, meta_filtered


# =====================================================================
# 2. NORMALIZATION
# =====================================================================

def normalize_error_trials(vsd_matrix, baseline_frames, std_pooled, eps=1e-8):
    """
    Normalize error-trial VSD data using the SAME std_pooled from
    correct-trial training. Per-trial baseline mean is computed fresh
    on the error trials themselves; std_pooled is reused, never recomputed.

    Parameters
        vsd_matrix : (pixels, frames, n_trials), FULL pixel space
        baseline_frames : (start, end) tuple, matching training config
        std_pooled : (pixels, 1, 1) array, from results["zscore_std_pooled"]

    Returns
        X_z : (pixels, frames, n_trials) normalized, full pixel space
            (apply ROI mask AFTER this)
    """
    X = np.asarray(vsd_matrix, dtype=float)
    std_pooled = np.asarray(std_pooled, dtype=float)

    if X.shape[0] != std_pooled.shape[0]:
        raise ValueError(
            f"Pixel dimension mismatch: vsd_matrix has {X.shape[0]} pixels, "
            f"std_pooled has {std_pooled.shape[0]} pixels. "
            "Did you accidentally ROI-mask before normalizing?"
        )

    start, end = baseline_frames
    baseline = X[:, start:end, :]
    mean_per_trial = baseline.mean(axis=1, keepdims=True)

    X_centered = X - mean_per_trial
    X_z = X_centered / np.maximum(std_pooled, eps)
    return X_z


# =====================================================================
# 3. DECODING
# =====================================================================

def decode_error_trials(X_error_roi, results):
    """
    Apply the final (100%-trained) sliding-window decoders to error trials.

    Returns dict with:
        centers, trial_pred (n_windows, n_trials), trial_score (n_windows, n_trials)
    """
    final_models = results["final_models"]
    centers = results["centers"]
    window_size = results["params"]["window_size"]

    n_trials = X_error_roi.shape[2]
    n_windows = len(final_models)

    trial_pred = np.zeros((n_windows, n_trials), dtype=int)
    trial_score = np.zeros((n_windows, n_trials), dtype=float)

    for w_idx, center in enumerate(centers):
        start = center - window_size // 2
        end = start + window_size
        X_win = X_error_roi[:, start:end, :]
        clf = final_models[w_idx]

        for trial_idx in range(n_trials):
            X_frames_trial = X_win[:, :, trial_idx].T
            scores = clf.decision_function(X_frames_trial)
            preds = (scores > 0).astype(int)
            trial_pred[w_idx, trial_idx] = np.bincount(preds).argmax()
            trial_score[w_idx, trial_idx] = scores.mean()

    return {"centers": centers, "trial_pred": trial_pred, "trial_score": trial_score}


def decode_error_trials_fold_averaged(X_error_roi, results, target_center):
    """
    Apply ALL fold-trained models (not just final_models) for one window
    to error trials, average scores across folds per trial. Since no fold
    ever saw the error trials during training, averaging across all folds
    is a legitimate, unbiased estimate (unlike for correct trials, where
    only the ONE fold that held out a given trial is unbiased for it).
    """
    centers = results["centers"]
    win_idx = np.where(centers == target_center)[0]
    if len(win_idx) == 0:
        raise ValueError(f"No window with center={target_center}")
    win_idx = win_idx[0]

    window_size = results["params"]["window_size"]
    start = target_center - window_size // 2
    end = start + window_size

    fold_models = results["fold_models_all"][win_idx]
    n_trials = X_error_roi.shape[2]
    n_folds = len(fold_models)

    fold_trial_scores = np.zeros((n_folds, n_trials))
    for f_idx, clf in enumerate(fold_models):
        for trial_idx in range(n_trials):
            X_frames_trial = X_error_roi[:, start:end, trial_idx].T
            fold_trial_scores[f_idx, trial_idx] = clf.decision_function(X_frames_trial).mean()

    fold_averaged_score = fold_trial_scores.mean(axis=0)
    return fold_averaged_score, fold_trial_scores


# =====================================================================
# 4. SESSION-LEVEL PIPELINE WRAPPERS
# =====================================================================


def process_error_trials_for_session(error_mat_path, results, ROI_mask_path, y_trials,
                                    face_cond, nonface_cond,
                                    Baseline_frames_zscore=(1, 24)):
    """
    Full error-trial pipeline for one session: load, filter to matched
    conditions, normalize, ROI mask, compute BOTH accuracy and
    label-consistent score comparisons across windows.

    Returns (acc_results, score_results, meta, X_error_roi)
    """
    vsd_matrix, meta = load_error_trials(error_mat_path)
    vsd_matrix, meta = select_conditions(vsd_matrix, meta, face_cond, nonface_cond)

    X_error_z = normalize_error_trials(vsd_matrix, Baseline_frames_zscore, results["zscore_std_pooled"])

    ROI_mask = np.load(ROI_mask_path).astype(bool)
    X_error_roi = X_error_z[ROI_mask, :, :]

    physical = meta["physicalLabel"]
    acc_results = compare_accuracy_across_windows(X_error_roi, results, physical, y_trials)
    score_results = compare_fold_averaged_across_windows(X_error_roi, results, physical, y_trials)

    return acc_results, score_results, meta, X_error_roi

# =====================================================================
# 5. STATISTICAL COMPARISONS (single session)
# =====================================================================

def compare_score_magnitude(correct_scores, error_scores, label="Score"):
    """
    Compare label-consistent decoder scores between correct and error trials.
    Inputs must already be sign-flipped (positive = matches trial's own true label).
    """
    correct_scores = np.asarray(correct_scores)
    error_scores = np.asarray(error_scores)

    n_correct, n_error = len(correct_scores), len(error_scores)
    mean_correct, mean_error = correct_scores.mean(), error_scores.mean()
    sd_correct, sd_error = correct_scores.std(ddof=1), error_scores.std(ddof=1)

    t_stat, p_welch = stats.ttest_ind(correct_scores, error_scores, equal_var=False)
    u_stat, p_mw = stats.mannwhitneyu(correct_scores, error_scores, alternative="two-sided")

    pooled_sd = np.sqrt(((n_correct-1)*sd_correct**2 + (n_error-1)*sd_error**2) / (n_correct+n_error-2))
    cohens_d = (mean_correct - mean_error) / pooled_sd

    print(f"--- {label} ---")
    print(f"Correct trials: n={n_correct}, mean={mean_correct:.3f}, SD={sd_correct:.3f}")
    print(f"Error trials:   n={n_error}, mean={mean_error:.3f}, SD={sd_error:.3f}")
    print(f"Welch's t-test: t={t_stat:.3f}, p={p_welch:.4f}")
    print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    return {
        "n_correct": n_correct, "n_error": n_error,
        "mean_correct": mean_correct, "mean_error": mean_error,
        "sd_correct": sd_correct, "sd_error": sd_error,
        "t_stat": t_stat, "p_welch": p_welch,
        "u_stat": u_stat, "p_mw": p_mw, "cohens_d": cohens_d,
    }


def compare_fold_averaged_across_windows(X_error_roi, results, physical, y_trials):
    """
    For every window: mean sign-flipped (label-consistent) score for
    correct trials (single held-out fold) vs. error trials (fold-averaged).
    """
    centers = results["centers"]
    n_windows = len(centers)
    window_size = results["params"]["window_size"]
    correct_sign_flip = np.where(y_trials == 1, 1, -1)
    error_sign_flip = np.where(physical == 1, 1, -1)

    n_correct, n_error = len(y_trials), len(physical)

    correct_mean_per_window = np.zeros(n_windows)
    error_mean_per_window = np.zeros(n_windows)
    correct_sem_per_window = np.zeros(n_windows)
    error_sem_per_window = np.zeros(n_windows)

    for w_idx, center in enumerate(centers):
        correct_raw = results["oof_trial_score"][w_idx, :]
        correct_consistent = correct_raw * correct_sign_flip
        correct_mean_per_window[w_idx] = correct_consistent.mean()
        correct_sem_per_window[w_idx] = correct_consistent.std(ddof=1) / np.sqrt(len(correct_consistent))

        fold_models = results["fold_models_all"][w_idx]
        start = center - window_size // 2
        end = start + window_size
        n_trials_err = X_error_roi.shape[2]
        fold_scores = np.zeros((len(fold_models), n_trials_err))
        for f_idx, clf in enumerate(fold_models):
            for trial_idx in range(n_trials_err):
                X_frames_trial = X_error_roi[:, start:end, trial_idx].T
                fold_scores[f_idx, trial_idx] = clf.decision_function(X_frames_trial).mean()
        error_raw = fold_scores.mean(axis=0)
        error_consistent = error_raw * error_sign_flip
        error_mean_per_window[w_idx] = error_consistent.mean()
        error_sem_per_window[w_idx] = error_consistent.std(ddof=1) / np.sqrt(len(error_consistent))

    print(f"Correct trials: n={n_correct}")
    print(f"Error trials:   n={n_error}")

    return {
        "centers": centers,
        "correct_mean": correct_mean_per_window, "correct_sem": correct_sem_per_window,
        "error_mean": error_mean_per_window, "error_sem": error_sem_per_window,
        "n_correct": n_correct, "n_error": n_error,
    }


def compare_accuracy_across_windows(X_error_roi, results, physical, y_trials):
    """
    Same structure as compare_fold_averaged_across_windows, but computes
    trial-level ACCURACY (fraction correct) instead of raw decision scores.
    """
    centers = results["centers"]
    n_windows = len(centers)
    window_size = results["params"]["window_size"]

    n_correct, n_error = len(y_trials), len(physical)

    correct_acc_per_window = np.zeros(n_windows)
    error_acc_per_window = np.zeros(n_windows)
    correct_sem_per_window = np.zeros(n_windows)
    error_sem_per_window = np.zeros(n_windows)

    for w_idx, center in enumerate(centers):
        correct_raw = results["oof_trial_score"][w_idx, :]
        correct_pred = (correct_raw > 0).astype(int)
        correct_is_right = (correct_pred == y_trials).astype(float)
        correct_acc_per_window[w_idx] = correct_is_right.mean()
        correct_sem_per_window[w_idx] = correct_is_right.std(ddof=1) / np.sqrt(len(correct_is_right))

        fold_models = results["fold_models_all"][w_idx]
        start = center - window_size // 2
        end = start + window_size
        n_trials_err = X_error_roi.shape[2]

        fold_correctness = np.zeros((len(fold_models), n_trials_err))
        for f_idx, clf in enumerate(fold_models):
            for trial_idx in range(n_trials_err):
                X_frames_trial = X_error_roi[:, start:end, trial_idx].T
                score = clf.decision_function(X_frames_trial).mean()
                pred = int(score > 0)
                fold_correctness[f_idx, trial_idx] = float(pred == physical[trial_idx])

        per_trial_acc = fold_correctness.mean(axis=0)
        error_acc_per_window[w_idx] = per_trial_acc.mean()
        error_sem_per_window[w_idx] = per_trial_acc.std(ddof=1) / np.sqrt(len(per_trial_acc))

    print(f"Correct trials: n={n_correct}")
    print(f"Error trials:   n={n_error}")

    return {
        "centers": centers,
        "correct_acc": correct_acc_per_window, "error_acc": error_acc_per_window,
        "correct_sem": correct_sem_per_window, "error_sem": error_sem_per_window,
        "n_correct": n_correct, "n_error": n_error,
    }


# =====================================================================
# 6. PLOTTING (single session)
# =====================================================================

def plot_error_trial_decoding(decode_results, meta):
    centers = decode_results["centers"]
    trial_score = decode_results["trial_score"]
    physical = meta["physicalLabel"]

    face_trials = physical == 1
    nonface_trials = physical == 0

    avg_face = trial_score[:, face_trials].mean(axis=1)
    sem_face = trial_score[:, face_trials].std(axis=1, ddof=1) / np.sqrt(face_trials.sum())
    avg_nonface = trial_score[:, nonface_trials].mean(axis=1)
    sem_nonface = trial_score[:, nonface_trials].std(axis=1, ddof=1) / np.sqrt(nonface_trials.sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, avg_face, color="tab:red",
            label=f"Physically FACE, reported NONFACE (n={face_trials.sum()})")
    ax.fill_between(centers, avg_face - sem_face, avg_face + sem_face, color="tab:red", alpha=0.2)
    ax.plot(centers, avg_nonface, color="tab:blue",
            label=f"Physically NONFACE, reported FACE (n={nonface_trials.sum()})")
    ax.fill_between(centers, avg_nonface - sem_nonface, avg_nonface + sem_nonface, color="tab:blue", alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Decoder decision score\n(+ = face-like, - = nonface-like)")
    ax.set_title("Decoder output on error trials, by physical stimulus category")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_match_fractions(decode_results, physical, behavioral):
    centers = decode_results["centers"]
    match_physical = (decode_results["trial_pred"] == physical[None, :]).mean(axis=1)
    match_behavioral = (decode_results["trial_pred"] == behavioral[None, :]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, match_physical, color="tab:green", label="Matches physical stimulus")
    ax.plot(centers, match_behavioral, color="tab:purple", label="Matches monkey's percept")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Fraction of error trials")
    ax.set_title("Decoder output on error trials: physical stimulus vs. percept")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_match_fractions_by_category(decode_results, physical, behavioral):
    centers = decode_results["centers"]
    trial_pred = decode_results["trial_pred"]

    face_trials = physical == 1
    nonface_trials = physical == 0
    n_face, n_nonface = face_trials.sum(), nonface_trials.sum()

    match_behavioral_face = (trial_pred[:, face_trials] == behavioral[face_trials][None, :]).mean(axis=1)
    match_behavioral_nonface = (trial_pred[:, nonface_trials] == behavioral[nonface_trials][None, :]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, match_behavioral_face, color="tab:red", label=f"Physically FACE trials (n={n_face})")
    ax.plot(centers, match_behavioral_nonface, color="tab:blue", label=f"Physically NONFACE trials (n={n_nonface})")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Fraction matching monkey's percept")
    ax.set_title("Decoder tracks percept, by physical stimulus category")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_trial_heatmap(decode_results, physical, meta):
    centers = decode_results["centers"]
    trial_pred = decode_results["trial_pred"]

    sort_idx = np.argsort(physical)
    sorted_pred = trial_pred[:, sort_idx]
    sorted_physical = physical[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sorted_pred.T, aspect="auto", cmap="coolwarm",
                    extent=[centers[0], centers[-1], 0, sorted_pred.shape[1]],
                    origin="lower", vmin=0, vmax=1)
    n_boundary = np.sum(sorted_physical == 0)
    ax.axhline(n_boundary, color="black", linewidth=1.5)
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Trial (sorted: nonface-physical below, face-physical above)")
    ax.set_title("Decoder guess per trial per window (blue=nonface, red=face)")
    plt.colorbar(im, ax=ax, label="Decoder guess (0=nonface, 1=face)")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_individual_traces(decode_results, physical):
    centers = decode_results["centers"]
    trial_score = decode_results["trial_score"]

    face_trials = physical == 1
    nonface_trials = physical == 0
    avg_face = trial_score[:, face_trials].mean(axis=1)
    avg_nonface = trial_score[:, nonface_trials].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(trial_score.shape[1]):
        color = "tab:red" if physical[i] == 1 else "tab:blue"
        ax.plot(centers, trial_score[:, i], color=color, alpha=0.25, linewidth=1)
    ax.plot(centers, avg_face, color="darkred", linewidth=2.5, label=f"Face-physical avg (n={face_trials.sum()})")
    ax.plot(centers, avg_nonface, color="darkblue", linewidth=2.5, label=f"Nonface-physical avg (n={nonface_trials.sum()})")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Decoder decision score")
    ax.set_title("Individual error trial trajectories with group averages")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_trial_dotplot(decode_results, physical, window_frames=(20, 50)):
    centers = decode_results["centers"]
    trial_score = decode_results["trial_score"]
    win_mask = (centers >= window_frames[0]) & (centers <= window_frames[1])
    avg_score = trial_score[win_mask, :].mean(axis=0)
    sign_flip = np.where(physical == 1, 1, -1)
    physical_consistent_score = avg_score * sign_flip

    face_trials = physical == 1
    nonface_trials = physical == 0

    fig, ax = plt.subplots(figsize=(7, 6))
    jitter_face = np.random.uniform(-0.05, 0.05, size=face_trials.sum())
    jitter_nonface = np.random.uniform(-0.05, 0.05, size=nonface_trials.sum())
    ax.scatter(np.zeros(face_trials.sum()) + jitter_face, physical_consistent_score[face_trials],
            color="tab:red", alpha=0.7, label=f"Face-physical (n={face_trials.sum()})")
    ax.scatter(np.ones(nonface_trials.sum()) + jitter_nonface, physical_consistent_score[nonface_trials],
            color="tab:blue", alpha=0.7, label=f"Nonface-physical (n={nonface_trials.sum()})")

    for x, mask, color in [(0, face_trials, "darkred"), (1, nonface_trials, "darkblue")]:
        m = physical_consistent_score[mask].mean()
        sem = physical_consistent_score[mask].std(ddof=1) / np.sqrt(mask.sum())
        ax.errorbar(x, m, yerr=sem, fmt="_", color=color, markersize=30,
                    markeredgewidth=3, capsize=8, linewidth=3, zorder=5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Face-physical", "Nonface-physical"])
    ax.set_ylabel("Score (+ = matches physical stimulus, - = matches percept)")
    ax.set_title(f"Per-trial decoder score, avg over frames {window_frames[0]}-{window_frames[1]}")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return fig, ax, physical_consistent_score


def plot_score_dotplot(score_matrix, labels, centers, window_frames, group_names=("Category 0", "Category 1")):
    win_mask = (centers >= window_frames[0]) & (centers <= window_frames[1])
    avg_score = score_matrix[win_mask, :].mean(axis=0)
    sign_flip = np.where(labels == 1, 1, -1)
    label_consistent_score = avg_score * sign_flip

    group0 = labels == 0
    group1 = labels == 1

    fig, ax = plt.subplots(figsize=(7, 6))
    jitter0 = np.random.uniform(-0.05, 0.05, size=group0.sum())
    jitter1 = np.random.uniform(-0.05, 0.05, size=group1.sum())
    ax.scatter(np.zeros(group0.sum()) + jitter0, label_consistent_score[group0],
            color="tab:blue", alpha=0.7, label=f"{group_names[0]} (n={group0.sum()})")
    ax.scatter(np.ones(group1.sum()) + jitter1, label_consistent_score[group1],
            color="tab:red", alpha=0.7, label=f"{group_names[1]} (n={group1.sum()})")

    for x, mask, color in [(0, group0, "darkblue"), (1, group1, "darkred")]:
        m = label_consistent_score[mask].mean()
        sem = label_consistent_score[mask].std(ddof=1) / np.sqrt(mask.sum())
        ax.errorbar(x, m, yerr=sem, fmt="_", color=color, markersize=30,
                    markeredgewidth=3, capsize=8, linewidth=3, zorder=5)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_names)
    ax.set_ylabel("Score (+ = matches own label)")
    ax.set_title(f"Per-trial decoder score, avg over frames {window_frames[0]}-{window_frames[1]}")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return fig, ax, label_consistent_score


def plot_fold_averaged_comparison(comparison_results, session_tag=None):
    centers = comparison_results["centers"]
    n_correct = comparison_results["n_correct"]
    n_error = comparison_results["n_error"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, comparison_results["correct_mean"], color="tab:green", label=f"Correct trials (n={n_correct})")
    ax.fill_between(centers,
                    comparison_results["correct_mean"] - comparison_results["correct_sem"],
                    comparison_results["correct_mean"] + comparison_results["correct_sem"],
                    color="tab:green", alpha=0.2)
    ax.plot(centers, comparison_results["error_mean"], color="tab:orange", label=f"Error trials (n={n_error})")
    ax.fill_between(centers,
                    comparison_results["error_mean"] - comparison_results["error_sem"],
                    comparison_results["error_mean"] + comparison_results["error_sem"],
                    color="tab:orange", alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Mean score (+ = matches own true label)")
    title = "Correct vs. error trials: label-consistent decoder confidence across windows"
    if session_tag:
        title = f"[{session_tag}] {title}"
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_accuracy_comparison(comparison_results, session_tag=None):
    centers = comparison_results["centers"]
    n_correct = comparison_results["n_correct"]
    n_error = comparison_results["n_error"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(centers, comparison_results["correct_acc"], color="tab:green", label=f"Correct trials (n={n_correct})")
    ax.fill_between(centers,
                    comparison_results["correct_acc"] - comparison_results["correct_sem"],
                    comparison_results["correct_acc"] + comparison_results["correct_sem"],
                    color="tab:green", alpha=0.2)
    ax.plot(centers, comparison_results["error_acc"], color="tab:orange", label=f"Error trials (n={n_error})")
    ax.fill_between(centers,
                    comparison_results["error_acc"] - comparison_results["error_sem"],
                    comparison_results["error_acc"] + comparison_results["error_sem"],
                    color="tab:orange", alpha=0.2)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Accuracy (matches trial's own true label)")
    ax.set_ylim(0, 1)
    title = "Correct vs. error trials: decoding accuracy across windows"
    if session_tag:
        title = f"[{session_tag}] {title}"
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


# =====================================================================
# 7. MULTI-SESSION GRAND AVERAGE
# =====================================================================

def grand_average_correct_and_error(correct_acc_list, error_acc_list, session_names, centers_ref):
    """
    Grand-average BOTH correct-trial and error-trial accuracy curves
    across sessions. Each session is one independent replicate.
    """
    for arr in correct_acc_list + error_acc_list:
        if len(arr) != len(centers_ref):
            raise ValueError("Mismatched window count across sessions -- check "
                            "start_frame/stop_frame/window_size consistency.")

    correct_stack = np.vstack(correct_acc_list)  # (n_sessions, n_windows)
    error_stack = np.vstack(error_acc_list)

    return {
        "centers": centers_ref,
        "grand_correct_mean": correct_stack.mean(axis=0),
        "grand_correct_sem": correct_stack.std(axis=0, ddof=1) / np.sqrt(correct_stack.shape[0]),
        "grand_error_mean": error_stack.mean(axis=0),
        "grand_error_sem": error_stack.std(axis=0, ddof=1) / np.sqrt(error_stack.shape[0]),
        "correct_stack": correct_stack, "error_stack": error_stack,
        "session_names": session_names, "n_sessions": correct_stack.shape[0],
    }


def plot_grand_average_correct_and_error(grand_results, chance_level=0.5, show_individual=True):
    centers = grand_results["centers"]
    n_sessions = grand_results["n_sessions"]

    fig, ax = plt.subplots(figsize=(9, 5))

    if show_individual:
        for i, name in enumerate(grand_results["session_names"]):
            ax.plot(centers, grand_results["correct_stack"][i, :], color="tab:green",
                    alpha=0.25, linewidth=1)
            ax.plot(centers, grand_results["error_stack"][i, :], color="tab:orange",
                    alpha=0.25, linewidth=1)

    ax.plot(centers, grand_results["grand_correct_mean"], color="darkgreen", linewidth=2.5,
            label=f"Correct trials, grand avg (n={n_sessions} sessions)")
    ax.fill_between(centers,
                    grand_results["grand_correct_mean"] - grand_results["grand_correct_sem"],
                    grand_results["grand_correct_mean"] + grand_results["grand_correct_sem"],
                    color="darkgreen", alpha=0.2)

    ax.plot(centers, grand_results["grand_error_mean"], color="darkorange", linewidth=2.5,
            label=f"Error trials, grand avg (n={n_sessions} sessions)")
    ax.fill_between(centers,
                    grand_results["grand_error_mean"] - grand_results["grand_error_sem"],
                    grand_results["grand_error_mean"] + grand_results["grand_error_sem"],
                    color="darkorange", alpha=0.2)

    ax.axhline(chance_level, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("Frame (window center)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Grand average across {n_sessions} sessions: correct vs. error trial decoding accuracy")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax