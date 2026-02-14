from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from functions_scripts.preprocessing_functions import green_gray_magenta
ourCmap = green_gray_magenta()


#------------------------------
#Data image plots
#------------------------------

def plot_superpixel_traces(data,xs = None,ys = None, nsubplots= 4,*,
                        overlay: bool = True,
                        title: str | None = None,
                        ylim: tuple[float, float] | None = None,
                        xlim: tuple[float, float] | None = None,
                        enable_popout: bool = True,
                        popout_double_click: bool = True,
                        popout_figsize: tuple[float, float] = (7, 4.5)):
    """
    Plot mean time-traces from spatial "superpixels" (binned blocks) of a 2D map.

    Input
    data : np.ndarray
        Shape:
        - (pixels, frames) or
        - (pixels, frames, conditions)
        Example: (10000, 120) where 10000 = 100*100.

    Binning
    The map is reshaped to (ys, xs, frames) and split into a grid of blocks.
    Block size is floor(xs/nsubplots) by floor(ys/nsubplots).
    Each subplot shows the mean trace of one block.

    Popout
    If enable_popout=True: double-click any subplot to open a larger figure
    with visible axes and tick labels.

    Returns
    binned : np.ndarray
        Shape (nplots, frames, nconds) where nplots = nrows*ncols.
    fig : matplotlib.figure.Figure
    axes : np.ndarray of Axes, shape (nrows, ncols)
    cid : callback id for the popout handler (or None)
    """
    X = np.asarray(data)
    if X.ndim == 2:
        X = X[:, :, None]  # (pixels, frames, 1)
    if X.ndim != 3:
        raise ValueError("data must be (pixels, frames) or (pixels, frames, conditions).")

    npix, nframes, nconds = X.shape

    # Infer xs, ys if not provided (must be a perfect square)
    if xs is None or ys is None:
        side = int(round(np.sqrt(npix)))
        if side * side != npix:
            raise ValueError(f"npix={npix} is not a perfect square. Provide xs and ys.")
        xs = side if xs is None else xs
        ys = side if ys is None else ys

    if xs * ys != npix:
        raise ValueError(f"pixels ({npix}) != xs*ys ({xs*ys}). Check xs/ys.")

    # MATLAB-style: pixperplot = floor(xs / nsubplots)
    xbin = int(xs // nsubplots)
    ybin = int(ys // nsubplots)
    if xbin < 1 or ybin < 1:
        raise ValueError("nsubplots is too large for this map size.")

    ncols = xs // xbin
    nrows = ys // ybin

    # Effective size (cropping like MATLAB fix())
    xs_eff = ncols * xbin
    ys_eff = nrows * ybin

    # -------- Vectorized binning (no loops over pixels) --------
    # Convert (pixels, frames, conds) -> (ys, xs, frames, conds)
    img = X.reshape(ys, xs, nframes, nconds)

    # Crop to multiples of bin size
    img = img[:ys_eff, :xs_eff, :, :]  # (ys_eff, xs_eff, frames, conds)

    # Block mean:
    # (nrows, ybin, ncols, xbin, frames, conds) -> mean over ybin & xbin
    b = img.reshape(nrows, ybin, ncols, xbin, nframes, nconds).mean(axis=(1, 3))
    # b shape: (nrows, ncols, frames, conds)

    # Flatten blocks into (nplots, frames, conds)
    binned = b.reshape(nrows * ncols, nframes, nconds)

    # -------- Plotting --------
    fig_w = max(6, 2.2 * ncols)
    fig_h = max(6, 2.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    if title:
        fig.suptitle(title, y=0.99)

    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            ax = axes[r, c]

            # Tiny-subplot style (like your MATLAB): no tick labels
            ax.set_xticks([])
            ax.set_yticks([])

            if overlay:
                for k in range(nconds):
                    ax.plot(binned[i, :, k], linewidth=1.0)
            else:
                ax.plot(binned[i, :, 0], linewidth=1.0)

            if ylim is not None:
                ax.set_ylim(*ylim)
            if xlim is not None:
                ax.set_xlim(*xlim)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # -------- Optional: double-click popout --------
    cid = None
    if enable_popout:
        ax_to_idx = {axes[r, c]: (r * ncols + c) for r in range(nrows) for c in range(ncols)}

        def _on_click(event):
            if popout_double_click and not getattr(event, "dblclick", False):
                return
            ax = event.inaxes
            if ax is None or ax not in ax_to_idx:
                return

            i = ax_to_idx[ax]
            rr, cc = divmod(i, ncols)

            pop_fig, pop_ax = plt.subplots(figsize=popout_figsize)
            pop_ax.set_title(f"Superpixel {i} (row={rr}, col={cc})")
            pop_ax.set_xlabel("Frame")
            pop_ax.set_ylabel("Activation")

            if overlay and nconds > 1:
                for k in range(nconds):
                    pop_ax.plot(binned[i, :, k], linewidth=1.6, label=f"cond {k}")
                pop_ax.legend(loc="best", frameon=False)
            else:
                pop_ax.plot(binned[i, :, 0], linewidth=1.8)

            # Make axes readable
            pop_ax.spines["top"].set_visible(False)
            pop_ax.spines["right"].set_visible(False)

            if ylim is not None:
                pop_ax.set_ylim(*ylim)
            if xlim is not None:
                pop_ax.set_xlim(*xlim)

            pop_fig.tight_layout()
            pop_fig.show()

        cid = fig.canvas.mpl_connect("button_press_event", _on_click)

    return binned, fig, axes, cid



def mimg(x, xsize=100, ysize=100, low='auto', high=None, frames=None, width=0):
    # if looking in raw data (not Zscored), data needs to substract 1  (-1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    m, n = x.shape
    if width <= 0:
        width = int(np.ceil(np.sqrt(n)))
    height = int(np.ceil(n / width))
    # Handle Clipping
    if isinstance(low, str) and low.lower() == 'auto':
        v_mins, v_maxs = [None] * n, [None] * n
    elif isinstance(low, str) and low.lower() == 'all':
        v_mins, v_maxs = [np.min(x)] * n, [np.max(x)] * n
    else:
        v_mins = np.full(n, low) if np.isscalar(low) else low
        v_maxs = np.full(n, high) if np.isscalar(high) else high
    fig, axes = plt.subplots(height, width, figsize=(width*3, height*3), squeeze=False)
    axes_flat = axes.flatten()
    for i in range(n):
        ax = axes_flat[i]
        # 1. Reshape using 'C' order because your data is "ordered by rows"
        # 2. We reshape as (xsize, ysize) or (ysize, xsize) depending on the source
        img_data = x[:, i].reshape((ysize, xsize), order='C')
        # Use 'jet' or 'nipy_spectral' 
        im = ax.imshow(img_data, cmap=ourCmap, vmin=v_mins[i], vmax=v_maxs[i], origin='upper')
        ax.axis('off')
        # Add frame numbers if provided
        if frames is not None:
            ax.set_title(f"frame {frames[i]}", fontsize=6)
    # Hide extra subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.tight_layout()
    return fig,axes_flat

#example usage
'''
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn1_270109b.npy")
x_avg = x.mean(axis=2)
x_avg_frames =  x_avg[:, 25:120]
fig,axes_flat =mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003,frames=range(25,120))
plt.show()
x = np.load(r"C:\project\vsdi-face-decoding\data\processed\condsXn\condsXn5_270109b.npy")
x_avg = x.mean(axis=2)
x_avg_frames =  x_avg[:, 25:120]
fig,axes_flat =mimg(x_avg_frames-1, xsize=100, ysize=100, low=-0.0009, high=0.003,frames=range(25,120))
plt.show()
'''




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


def plot_frame_vs_trial_bars(results, chance= 0.5, title: str = "Frame vs Trial accuracy per outer fold",
                            figsize=(8, 4.5), ylim=(0.4, 1.0), show_mean_sem= True):
    """
    Side-by-side bars per outer fold: frame accuracy vs trial (majority vote) accuracy.

    Expects `results` from run_nested_cv_selectC_then_eval with:
    results["outer_folds"][i]["acc"]       (frame accuracy)
    results["outer_folds"][i]["acc_trial"] (trial accuracy)

    Plots:
    - grouped bars per fold
    - chance line
    - optional mean ± SEM markers for each series
    """
    if not isinstance(results, dict) or "outer_folds" not in results:
        raise ValueError("results must be a dict with key 'outer_folds' (output of nested CV).")

    folds = results["outer_folds"]
    if len(folds) == 0:
        raise ValueError("No outer folds found in results['outer_folds'].")

    acc_frame = np.asarray([f["acc"] for f in folds], dtype=float)
    acc_trial = np.asarray([f["acc_trial"] for f in folds], dtype=float)

    k = acc_frame.size
    x = np.arange(k)
    width = 0.38

    # Means / SEM
    mean_f = float(np.nanmean(acc_frame))
    mean_t = float(np.nanmean(acc_trial))
    sem_f = float(np.nanstd(acc_frame, ddof=1) / np.sqrt(k)) if k > 1 else np.nan
    sem_t = float(np.nanstd(acc_trial, ddof=1) / np.sqrt(k)) if k > 1 else np.nan

    plt.figure(figsize=figsize)
    plt.bar(x - width/2, acc_frame, width=width, alpha=0.85, label="Frame acc")
    plt.bar(x + width/2, acc_trial, width=width, alpha=0.85, label="Trial acc (vote)")
    plt.axhline(chance, linestyle="--", linewidth=2, label="Chance")

    # Optional mean ± SEM markers (placed to the right)
    if show_mean_sem:
        mean_x = k + 0.6
        # Frame mean
        if np.isfinite(sem_f):
            plt.errorbar(mean_x - 0.12, mean_f, yerr=sem_f, fmt="o", capsize=6, linewidth=2)
            plt.text(mean_x - 0.12, mean_f + 0.02, f"{mean_f:.2f}±{sem_f:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        else:
            plt.plot([mean_x - 0.12], [mean_f], "o")
            plt.text(mean_x - 0.12, mean_f + 0.02, f"{mean_f:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Trial mean
        if np.isfinite(sem_t):
            plt.errorbar(mean_x + 0.12, mean_t, yerr=sem_t, fmt="o", capsize=6, linewidth=2)
            plt.text(mean_x + 0.12, mean_t + 0.02, f"{mean_t:.2f}±{sem_t:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        else:
            plt.plot([mean_x + 0.12], [mean_t], "o")
            plt.text(mean_x + 0.12, mean_t + 0.02, f"{mean_t:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Extend x-axis ticks to include "Mean"
        plt.xticks(
            list(x) + [mean_x],
            [f"Fold {i+1}" for i in range(k)] + ["Mean"],
            fontsize=12
        )
        plt.xlim(-0.6, mean_x + 0.8)
    else:
        plt.xticks(x, [f"Fold {i+1}" for i in range(k)], fontsize=12)

    plt.ylabel("Accuracy", fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    plt.show()




#############################################

def plot_metric_hist(fold_values,chance= 0.5,
                    title= "Metric distribution",
                    xlabel = "Metric value",
                    figsize=(7, 4),
                    bins = 30,
                    xlim=(0, 1)):
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


