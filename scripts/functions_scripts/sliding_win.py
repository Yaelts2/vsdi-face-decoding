import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from functions_scripts import feature_extraction as fe
from functions_scripts import ml_cv as cv





def sliding_window_decode_with_stats(X_pix_frames_trials,   # (pixels, frames, trials)  e.g. (8518, 256, 56)
                                    y_trials,               # (trials,)
                                    make_estimator,         # callable -> fresh estimator each fold
                                    window_size=5,
                                    start_frame=15,
                                    stop_frame=125,         # stop_frame
                                    step=1,
                                    n_splits=5,
                                    return_fold_weights=False):
    """
    Sliding window decoding with:
    - frame-level accuracy mean+SEM across folds
    - trial-level accuracy mean+SEM across folds (majority vote across frames in trial)
    - per-window mean weights across folds

    Returns dict with arrays over windows.
    """
    X = np.asarray(X_pix_frames_trials)
    y_trials = np.asarray(y_trials).astype(int)

    n_pixels, n_frames, n_trials = X.shape
    if y_trials.shape[0] != n_trials:
        raise ValueError(f"y_trials has {y_trials.shape[0]} but X has {n_trials} trials")

    stop_frame = min(int(stop_frame), n_frames)
    last_start = min(n_frames - window_size, stop_frame - window_size)
    if last_start < start_frame:
        raise ValueError("stop_frame is too early for the given start_frame/window_size")

    gkf = GroupKFold(n_splits=n_splits)

    centers = []
    frame_acc_mean, frame_acc_sem = [], []
    trial_acc_mean, trial_acc_sem = [], []
    w_mean_windows = []
    w_sem_windows = []  

    # optional storage
    fold_frame_acc = []
    fold_trial_acc = []
    fold_weights_all = [] if return_fold_weights else None
    # Loop over sliding windows
    for start in range(start_frame, last_start + 1, step):
        end = start + window_size
        center = start + window_size // 2
        # Slice window: (pixels, window_size, trials)
        X_win = X[:, start:end, :]
        # Convert to frames-as-samples: X_frames (trials*window, pixels)
        X_frames, y_frames, groups = fe.frames_as_samples(X_win, y_trials,
                                                        trial_axis=2, frame_axis=1, pixel_axis=0)
        fold_acc_f = []
        fold_acc_t = []
        W_folds = []
        # kfold for a single window
        for tr_idx, te_idx in gkf.split(X_frames, y_frames, groups):
            clf = make_estimator()
            clf.fit(X_frames[tr_idx], y_frames[tr_idx])
            y_pred = clf.predict(X_frames[te_idx])

            # (1) frame-level accuracy
            acc_f = accuracy_score(y_frames[te_idx], y_pred)
            fold_acc_f.append(float(acc_f))

            # (2) trial-level accuracy by majority vote across frames per trial
            y_true_trial, y_pred_trial= cv.majority_vote_trial_predictions(y_pred,
                                                                        y_frames[te_idx],
                                                                        groups[te_idx])
            acc_t = accuracy_score(y_true_trial, y_pred_trial)
            fold_acc_t.append(float(acc_t))

            # (3) weights
            w = cv.extract_linear_weights_general(clf)
            if w is None:
                raise ValueError("Estimator does not expose linear weights (coef_).")
            W_folds.append(w)

        W_folds = np.vstack(W_folds)  # (n_folds, n_features)
        centers.append(center)
        frame_acc_mean.append(float(np.mean(fold_acc_f)))
        frame_acc_sem.append(_sem(fold_acc_f))
        trial_acc_mean.append(float(np.mean(fold_acc_t)))
        trial_acc_sem.append(_sem(fold_acc_t))

        w_mean_windows.append(np.mean(W_folds, axis=0))
        w_sem_windows.append(np.std(W_folds, axis=0, ddof=1) if W_folds.shape[0] > 1 else np.full(W_folds.shape[1], np.nan))

        fold_frame_acc.append(fold_acc_f)
        fold_trial_acc.append(fold_acc_t)
        if return_fold_weights:
            fold_weights_all.append(W_folds)

    out = {
        "centers": np.asarray(centers),
        "frame_acc_mean": np.asarray(frame_acc_mean),
        "frame_acc_sem":  np.asarray(frame_acc_sem),
        "trial_acc_mean": np.asarray(trial_acc_mean), # (n_windows,)
        "trial_acc_sem":  np.asarray(trial_acc_sem),
        "w_mean_windows": np.vstack(w_mean_windows),  # (n_windows, n_features)
        "w_sem_windows":  np.vstack(w_sem_windows),   # (n_windows, n_features) diagnostic
        "fold_frame_acc": np.asarray(fold_frame_acc), # (n_windows, n_splits)
        "fold_trial_acc": np.asarray(fold_trial_acc), # (n_windows, n_splits)
        "params": {
            "window_size": window_size,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "step": step,
            "n_splits": n_splits,
        }
    }
    if return_fold_weights:
        out["W_folds_per_window"] = fold_weights_all  # list length n_windows, each (n_splits, n_features)
    return out


def plot_sliding_window_accuracy_with_sem(res: dict,
                                        chance: float = 0.5,
                                        title: str = "Sliding-window decoding accuracy",
                                        xlabel: str = "Frame (window center)",
                                        ylabel: str = "Accuracy",
                                        ylim=(0.4, 1.0),
                                        figsize=(8, 4),
                                        show_points: bool = True):
    """
    Plot frame-level and trial-level accuracy across sliding windows with SEM shading.

    Parameters
    ----------
    res : dict
        Output from sliding_window_decode_with_stats / sliding_window_decode_with_stats-like function.
        Must contain:
        - centers
        - frame_acc_mean, frame_acc_sem
        - trial_acc_mean, trial_acc_sem
    chance : float
        Chance accuracy line (e.g., 0.5 for binary).
    """

    centers = np.asarray(res["centers"])
    f_mean  = np.asarray(res["frame_acc_mean"])
    f_sem   = np.asarray(res["frame_acc_sem"])
    t_mean  = np.asarray(res["trial_acc_mean"])
    t_sem   = np.asarray(res["trial_acc_sem"])

    plt.figure(figsize=figsize)

    # Frame-level curve + SEM
    plt.plot(centers, f_mean, label="Frame-level accuracy")
    plt.fill_between(centers, f_mean - f_sem, f_mean + f_sem, alpha=0.2)

    # Trial-level curve + SEM
    plt.plot(centers, t_mean, label="Trial-level accuracy (majority vote)")
    plt.fill_between(centers, t_mean - t_sem, t_mean + t_sem, alpha=0.2)

    # Optional points
    if show_points:
        plt.scatter(centers, f_mean, s=12)
        plt.scatter(centers, t_mean, s=12)

    # Chance line
    if chance is not None:
        plt.axhline(chance, linestyle="--", linewidth=1, label=f"Chance = {chance:g}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    plt.show()





