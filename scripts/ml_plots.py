# ml_plots.py
# Plotting utilities compatible with the cv framework in ml_cv.py
#
# Design goals:
# - No training / CV logic here (only plotting)
# - Works with both:
#     * single-layer run_cv outputs (results["folds"], results["oof_*"])
#     * nested outputs (nested["outer_folds"], nested["oof_*"])
# - Weight plots take weight vectors (w) instead of estimator objects (Pipeline-safe)
#
# Requirements:
# - feature_extraction.mimg must exist (your VSDI image helper)

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from feature_extraction import mimg


# -----------------------------
# Small adapters (results -> arrays)
# -----------------------------

def get_fold_values(results: dict, key: str = "acc") -> np.ndarray:
    """
    Extract per-fold values from either:
    - results["folds"] (single-layer CV)
    - results["outer_folds"] (nested CV)
    """
    if isinstance(results, dict) and "folds" in results:
        return np.asarray([f[key] for f in results["folds"]], dtype=float)
    if isinstance(results, dict) and "outer_folds" in results:
        return np.asarray([f[key] for f in results["outer_folds"]], dtype=float)
    raise ValueError("Expected results dict with 'folds' or 'outer_folds'.")


def get_oof_arrays(results: dict):
    """
    Return (y_true, y_pred, y_scores, mask_scores) from results.
    Works with both single-layer and nested outputs (we standardized keys).
    """
    required = ("oof_y_true", "oof_pred", "oof_scores", "oof_has_score")
    for k in required:
        if k not in results:
            raise ValueError(f"results missing key '{k}'. "
                            f"Did you run with expect_full_coverage=True (run_cv) "
                            f"or use nested CV output?")
    y_true = np.asarray(results["oof_y_true"], dtype=int).ravel()
    y_pred = np.asarray(results["oof_pred"], dtype=int).ravel()
    y_scores = np.asarray(results["oof_scores"], dtype=float).ravel()
    mask = np.asarray(results["oof_has_score"], dtype=bool).ravel()
    return y_true, y_pred, y_scores, mask


# -----------------------------
# Accuracy plots
# -----------------------------

def plot_accuracy_kfold_bars(
    fold_acc,
    chance: float = 0.5,
    title: str = "Accuracy per fold",
    figsize=(6, 4),
    ylim=(0.4, 1.0),
):
    """
    Bar plot of fold accuracies + mean ± SEM + chance line.

    fold_acc can be:
    - list/array of accuracies
    - results dict from run_cv / nested CV
    """
    if isinstance(fold_acc, dict):
        acc = get_fold_values(fold_acc, key="acc")
    else:
        acc = np.asarray(fold_acc, dtype=float)

    k = acc.size
    if k == 0:
        raise ValueError("No fold accuracies to plot.")

    mean = float(np.nanmean(acc))
    sem = float(np.nanstd(acc, ddof=1) / np.sqrt(k)) if k > 1 else np.nan

    x = np.arange(1, k + 1)

    plt.figure(figsize=figsize)
    plt.bar(x, acc, alpha=0.8)

    plt.axhline(chance, linestyle="--", linewidth=2, label="Chance")
    plt.axhline(mean, linestyle=":", linewidth=2, label="Mean")

    # mean ± SEM marker
    mean_x = k + 0.8
    if np.isfinite(sem):
        plt.errorbar(mean_x, mean, yerr=sem, fmt="o", capsize=6, linewidth=2)
        plt.text(
            mean_x, mean + 0.02,
            f"{mean:.2f} ± {sem:.2f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )
    else:
        plt.plot([mean_x], [mean], "o")
        plt.text(mean_x, mean + 0.02, f"{mean:.2f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

    plt.xticks(list(x) + [mean_x],
               [f"Fold {i}" for i in x] + ["Mean"],
               fontsize=12)

    plt.ylabel("Accuracy", fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_metric_hist(
    fold_values,
    chance: float = 0.5,
    title: str = "Metric distribution",
    xlabel: str = "Metric value",
    figsize=(7, 4),
    bins: int = 30,
    xlim=(0, 1),
):
    """
    Histogram of many-fold distributions (e.g., LOPO with many folds).

    fold_values can be:
      - list/array of values
      - results dict (will use acc by default)
    """
    if isinstance(fold_values, dict):
        v = get_fold_values(fold_values, key="acc")
    else:
        v = np.asarray(fold_values, dtype=float)

    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        raise ValueError("No finite values to plot.")

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else np.nan

    plt.figure(figsize=figsize)
    plt.hist(v, bins=bins, alpha=0.85)

    plt.axvline(chance, linestyle="--", linewidth=2, label="Chance")
    plt.axvline(mean, linestyle=":", linewidth=2, label="Mean")

    txt = f"mean={mean:.3f} ± {std:.3f} (SD)\nN={n}"
    plt.text(
        0.98, 0.98, txt,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
    )

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.title(title, fontsize=16, fontweight="bold")
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Confusion matrix + ROC (OOF-based)
# -----------------------------

def plot_confusion_matrix(
    y_true=None,
    y_pred=None,
    results: dict | None = None,
    class_names=("Non-face", "Face"),
    title="Confusion matrix",
    figsize=(5.5, 4.8),
    show_counts=True,
):
    """
    Row-normalized confusion matrix with optional count overlay.
    Use either (y_true, y_pred) OR pass results dict containing oof arrays.
    """
    if results is not None:
        y_true, y_pred, _, _ = get_oof_arrays(results)

    if y_true is None or y_pred is None:
        raise ValueError("Provide (y_true, y_pred) or results dict.")

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    cm_counts = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm = np.divide(cm_counts, np.maximum(row_sums, 1), dtype=float)

    plt.figure(figsize=figsize)
    im = plt.imshow(cm, vmin=0, vmax=1)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Proportion", fontsize=12)

    n = len(class_names)
    plt.xticks(np.arange(n), class_names, fontsize=12)
    plt.yticks(np.arange(n), class_names, fontsize=12)
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            txt = f"{val:.2f}"
            if show_counts:
                txt += f"\n(n={cm_counts[i, j]})"
            plt.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if val >= 0.5 else "black",
            )

    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true=None,
    y_scores=None,
    results: dict | None = None,
    title="ROC curve",
    figsize=(5.5, 4.8),
):
    """
    Plot ROC curve with AUC.
    Use either (y_true, y_scores) OR pass results dict containing oof arrays.

    Notes:
    - If using results dict, we automatically mask to finite scores via oof_has_score.
    """
    if results is not None:
        y_true, _, y_scores, mask = get_oof_arrays(results)
        y_true = y_true[mask]
        y_scores = y_scores[mask]

    if y_true is None or y_scores is None:
        raise ValueError("Provide (y_true, y_scores) or results dict.")

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    if y_true.size == 0:
        raise ValueError("No samples available for ROC (all scores were missing).")
    if np.unique(y_true).size < 2:
        raise ValueError("ROC requires both classes present in y_true.")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color="gray", label="Chance")

    plt.xlabel("False positive rate", fontsize=14)
    plt.ylabel("True positive rate", fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=12, frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fpr, tpr, float(roc_auc)



def plot_groupkfold_splits(y, groups, splitter, ax=None, title="CV splits"):
    """
    Visualize CV splits (train/test) for any splitter that supports .split().

    Rows:
    - 0..n_folds-1 : test assignment (1=test, 0=train)
    - class row    : y (normalized to [0,1] for display)
    - group row    : groups (normalized to [0,1] for display)

    Parameters
    ----------
    y : array-like, shape (n_samples,)
    groups : array-like, shape (n_samples,)
        Trial IDs in your case.
    splitter : object
        Any sklearn-like splitter with .split(X, y, groups) or iterable of splits.
    ax : matplotlib axis, optional
    title : str

    Returns
    -------
    ax : matplotlib axis
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    n_samples = y.shape[0]
    if groups.shape[0] != n_samples:
        raise ValueError("groups must have same length as y")

    # Create dummy X only for splitters that require X
    X_dummy = np.zeros((n_samples, 1), dtype=float)

    # Use your generic splitter handler if you want:
    # splits = get_splits(splitter, X_dummy, y, groups)
    # Otherwise keep it explicit:
    splits = list(splitter.split(X_dummy, y, groups))

    if len(splits) == 0:
        raise ValueError("Splitter produced 0 splits")

    # Build split matrix: 0=train, 1=test
    split_mat = np.zeros((len(splits), n_samples), dtype=int)
    for i, (_, te) in enumerate(splits):
        split_mat[i, np.asarray(te, dtype=int)] = 1

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.2))

    # Use default colormap (avoid hard-coded colors unless needed)
    ax.imshow(split_mat, aspect="auto", interpolation="nearest")

    n_folds = split_mat.shape[0]

    # Helper: normalize to [0,1] safely
    def _norm01(v):
        v = np.asarray(v)
        if v.dtype.kind not in ("i", "u", "f"):
            # non-numeric groups: factorize
            _, v = np.unique(v, return_inverse=True)
        v = v.astype(float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        denom = (vmax - vmin)
        if not np.isfinite(denom) or denom == 0:
            return np.zeros_like(v, dtype=float)
        return (v - vmin) / denom

    y_row = _norm01(y).reshape(1, -1)
    g_row = _norm01(groups).reshape(1, -1)

    # Put y and group rows beneath the split matrix
    ax.imshow(y_row, aspect="auto", interpolation="nearest",
            extent=(0, n_samples, n_folds, n_folds + 1))
    ax.imshow(g_row, aspect="auto", interpolation="nearest",
            extent=(0, n_samples, n_folds + 1, n_folds + 2))

    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Fold")

    ax.set_yticks(list(range(n_folds)) + [n_folds + 0.5, n_folds + 1.5])
    ax.set_yticklabels([str(i) for i in range(n_folds)] + ["y", "group"])

    # Put fold 0 at top
    ax.set_ylim(n_folds + 2, -0.5)

    return ax


