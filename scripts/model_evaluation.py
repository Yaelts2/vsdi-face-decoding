# model_evaluation.py

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from feature_extraction import mimg
from sklearn.metrics import roc_curve, auc


def plot_accuracy_kfold_bars(cv_fold_acc,
                            chance=0.5,
                            title="Accuracy per fold ",
                            figsize=(6, 4),
                            ):
    """
    Bar plot of each CV fold accuracy + mean ± SEM.

    - Individual bars per fold
    - Mean ± SEM shown separately
    - Chance level indicated
    - Optimized for presentation (larger fonts)
    """
    acc = np.asarray(cv_fold_acc, dtype=float)
    k = acc.size
    mean = np.nanmean(acc)
    sem = np.nanstd(acc, ddof=1) / np.sqrt(k) if k > 1 else 0.0
    x = np.arange(1, k + 1)
    plt.figure(figsize=figsize)
    # Fold bars
    plt.bar(x, acc, alpha=0.8)
    # Reference lines
    plt.axhline(chance, linestyle="--", linewidth=2, label="Chance")
    plt.axhline(mean, linestyle=":", linewidth=2, label="Mean")
    # Mean ± SEM point
    mean_x = k + 0.8
    plt.errorbar(mean_x,
                mean,
                yerr=sem,
                fmt="o",
                capsize=6,
                linewidth=2,
                )
    # Mean annotation
    plt.text(mean_x,
            mean + 0.02,
            f"{mean:.2f} ± {sem:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            )

    # Axes formatting
    plt.xticks(list(x) + [mean_x],
            [f"Fold {i}" for i in x] + ["Mean"],
            fontsize=12,
            )
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.ylim(0.4, 1.0)
    plt.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(y_true,
                        y_pred,
                        class_names=("Non-face", "Face"),
                        title="Confusion matrix",
                        figsize=(5.5, 4.8),
                        show_counts=True):
    """
    Clean row-normalized confusion matrix (per true class).
    - Fixed color scale (0–1) for consistency across runs
    - Larger typography for slides
    - Optional raw counts overlay (useful for imbalanced folds)
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    cm_counts = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm = np.divide(cm_counts, np.maximum(row_sums, 1), dtype=float)  # avoid /0
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
    # Cell labels with contrast-aware text color
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            txt = f"{val:.2f}"
            if show_counts:
                txt += f"\n(n={cm_counts[i, j]})"
            plt.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if val >= 0.5 else "black",
            )
    plt.tight_layout()
    plt.show()




def plot_roc_curve(y_true,
                y_scores,
                title="ROC curve",
                figsize=(5.5, 4.8)):
    """
    Plot ROC curve with AUC.
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Binary ground-truth labels (0/1)
    y_scores : array-like, shape (n_samples,)
        Continuous decision values (e.g., SVM decision_function)
    """

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    # ROC curve
    plt.plot(fpr,tpr,
            linewidth=2.5,
            label=f"AUC = {roc_auc:.2f}")
    # Chance diagonal
    plt.plot([0, 1],[0, 1],
            linestyle="--",
            linewidth=2,
            color="gray",
            label="Chance")

    plt.xlabel("False positive rate", fontsize=14)
    plt.ylabel("True positive rate", fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=12, frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fpr, tpr, roc_auc

    

def plot_spatial_weights(best_model,
                        ROI_mask,
                        n_frames,
                        xsize=100,
                        ysize=100,
                        aggregate="sum",
                        title="Spatial decoding weights",
                        figsize=(6.5, 6.5)):
    """
    Project linear SVM weights back into cortical space.
    Parameters
    ----------
    best_model : trained linear SVM
        Must have coef_ attribute
    ROI_mask : array-like, bool
        Flat mask of shape (n_pixels,)
    n_frames : int
        Number of temporal features per pixel
    aggregate : {"sum", "mean"}
        How to aggregate absolute weights across time
    """
    ROI_mask = np.asarray(ROI_mask, dtype=bool).ravel()
    # --- reshape weights ---
    coef = best_model.coef_.ravel()
    expected_size = ROI_mask.sum() * n_frames
    if coef.size != expected_size:
        raise ValueError(
            f"coef_ size ({coef.size}) does not match "
            f"ROI pixels × n_frames ({expected_size})"
        )
    weights = coef.reshape(ROI_mask.sum(), n_frames)
    # --- temporal aggregation ---
    if aggregate == "sum":
        spatial_map_flat = np.sum(np.abs(weights), axis=1)
        agg_label = "Σ |w(t)|"
    elif aggregate == "mean":
        spatial_map_flat = np.mean(np.abs(weights), axis=1)
        agg_label = "mean |w(t)|"
    else:
        raise ValueError("aggregate must be 'sum' or 'mean'")
    # --- project back to image space ---
    spatial_weight_img = np.zeros(ROI_mask.shape, dtype=float)
    spatial_weight_img[ROI_mask] = spatial_map_flat
    # scale only by ROI values
    vmin = spatial_map_flat.min()
    vmax = spatial_map_flat.max()
    # --- plot ---
    mimg(spatial_weight_img,
        xsize=xsize,
        ysize=ysize,
        low=vmin,
        high=vmax)

    plt.title(f"{title}",fontsize=16,fontweight="bold")
    plt.tight_layout()
    plt.show()
    return spatial_weight_img



def plot_top_weights(spatial_weight_img, percentile=95, xsize=100, ysize=100, title="Top 5% weights"):
    """
    Takes an existing spatial weight map and plots only the pixels 
    above the specified percentile.
    """
    # 1. Isolate the non-zero values (the ROI pixels) to get a fair percentile
    roi_pixels = spatial_weight_img[spatial_weight_img > 0]
    if len(roi_pixels) == 0:
        print("Warning: The spatial_weight_img is empty or contains only zeros.")
        return
    # 2. Calculate the threshold value (e.g., 95th percentile)
    thresh_value = np.percentile(roi_pixels, percentile)
    # 3. Create the thresholded image
    # We keep the original values for pixels above threshold, set others to 0
    thresholded_img = np.where(spatial_weight_img >= thresh_value, spatial_weight_img, 0)
    # 4. Plot using your existing mimg function
    # We set low=0 so the background is neutral and high to the max of the map
    mimg(thresholded_img,
        xsize=xsize,
        ysize=ysize,
        low=0, 
        high=spatial_weight_img.max())

    plt.title(f"{title}\n(Top {100-percentile}%)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return thresholded_img


def permutation_test_linear_svm_fast(
    X, y,
    real_acc=None,
    n_perm=100,
    n_splits=5,
    C=0.01,
    seed=0,
    print_every=10,
    bins=30,
    title="Permutation test (Linear SVM)",
    figsize=(6.2, 4.8),
):
    """
    Permutation test for decoding accuracy using a LinearSVC + fixed CV splits.

    Returns
    -------
    shuffled_acc : (n_perm,) array
        Mean CV accuracy for each label permutation.
    p_value : float or None
        One-sided permutation p-value: P(shuffled_acc >= real_acc).
        Uses +1 smoothing: (k+1)/(n+1).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int).ravel()
    X = np.asarray(X)
    # Precompute CV splits ONCE (keeps the test fair & fast)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))
    shuffled_acc = np.empty(n_perm, dtype=float)
    t0 = time.time()
    for p in range(n_perm):
        y_shuff = rng.permutation(y)
        fold_acc = np.empty(len(splits), dtype=float)
        for i, (train_idx, test_idx) in enumerate(splits):
            clf = LinearSVC(C=C, dual=True, max_iter=5000, random_state=seed)
            clf.fit(X[train_idx], y_shuff[train_idx])
            y_pred = clf.predict(X[test_idx])
            fold_acc[i] = accuracy_score(y_shuff[test_idx], y_pred)
        shuffled_acc[p] = float(np.mean(fold_acc))
        if print_every and (p + 1) % print_every == 0:
            elapsed = time.time() - t0
            per_perm = elapsed / (p + 1)
            eta = per_perm * (n_perm - (p + 1))
            print(
                f"perm {p+1}/{n_perm} | elapsed {elapsed:.1f}s | "
                f"~{per_perm:.2f}s/perm | ETA {eta/60:.1f} min"
            )
    # p-value (one-sided) with +1 smoothing
    p_value = None
    if real_acc is not None:
        real_acc = float(real_acc)
        p_value = (np.sum(shuffled_acc >= real_acc) + 1) / (n_perm + 1)
    # --- Plot (presentation-ready) ---
    plt.figure(figsize=figsize)
    plt.hist(shuffled_acc, bins=bins, alpha=0.75, edgecolor="black", linewidth=1)
    # Mark chance (optional but useful) and real performance
    chance = 1.0 / len(np.unique(y))
    plt.axvline(chance, linestyle="--", linewidth=2, color="gray", label=f"Chance = {chance:.2f}")
    if real_acc is not None:
        plt.axvline(real_acc, linewidth=3, label=f"Real acc = {real_acc:.3f}")
        if p_value is not None:
            plt.text(0.02, 0.95,f"p = {p_value:.4f}", transform=plt.gca().transAxes, ha="left", va="top", fontsize=13,fontweight="bold")
    plt.xlabel("Shuffled mean CV accuracy", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11, frameon=False)
    plt.tight_layout()
    plt.show()

    return shuffled_acc, p_value
