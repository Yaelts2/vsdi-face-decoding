import numpy as np
import matplotlib.pyplot as plt
import math
from preprocessing_functions import green_gray_magenta

ourCmap = green_gray_magenta()





def weights_to_roi_image(w_roi,roi_mask_flat,pixels= 100, fill_value= np.nan,) -> np.ndarray:
    """
    Map a 1D weight vector defined on ROI pixels back into full image space.

    Parameters
    ----------
    w_roi : array-like, shape (n_roi_pixels,)
        Weights for ROI pixels only (the feature space you trained on).
    roi_mask_flat : array-like, shape (pixels*pixels,) or (pixels*pixels, 1)
        Boolean mask marking ROI pixels in the full image (flattened).
        True where ROI pixel exists.
    pixels : int
        Image side length (100 -> returns 100x100).
    fill_value : float
        Value to assign outside ROI (default NaN).

    Returns
    -------
    img : np.ndarray, shape (pixels, pixels)
        Full image where ROI pixels contain weights and outside-ROI is fill_value.
    """
    w_roi = np.asarray(w_roi, dtype=float).ravel()
    mask = np.asarray(roi_mask_flat, dtype=bool).ravel()

    if mask.size != pixels * pixels:
        raise ValueError(f"roi_mask_flat must have size {pixels*pixels}, got {mask.size}")

    n_roi = int(mask.sum())
    if w_roi.size != n_roi:
        raise ValueError(f"ROI has {n_roi} pixels but w_roi has {w_roi.size} weights")

    full = np.full(pixels * pixels, fill_value, dtype=float)
    full[mask] = w_roi
    return full.reshape(pixels, pixels)


def draw_weight_map(ax, img, cmap= ourCmap, clip= "sym", title= None,
                    show_colorbar= True):
    """
    Draw a 2D weight map on an existing axis (no figure creation, no show).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    img : np.ndarray, shape (pixels, pixels)
        Weight image (output of weights_to_roi_image).
    cmap : str
        Colormap.
    clip : tuple or "sym" or None
        - "sym": symmetric clipping around zero
        - (vmin, vmax): manual clipping
        - None: no clipping
    title : str or None
        Optional title.
    show_colorbar : bool
        Whether to add a colorbar to this axis.

    Returns
    -------
    im : matplotlib.image.AxesImage
        Image handle (useful for shared colorbars).
    """
    img = np.asarray(img, dtype=float)

    if clip == "sym":
        vmax = np.nanmax(np.abs(img))
        vmin = -vmax
    elif clip is None:
        vmin = vmax = None
    else:
        vmin, vmax = clip

    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return im



def plot_all_fold_weight_maps(W_outer,roi_mask_flat, pixels = 100,n_cols = 5,
                            cmap= ourCmap, clip= (-0.0002, 0.0002),figsize_per_panel= 3.0):
    """
    Plot all outer-fold weight maps in a grid.

    Parameters
    ----------
    W_outer : np.ndarray, shape (n_folds, n_features)
        Per-fold weight vectors.
    roi_mask_flat : np.ndarray, shape (pixels*pixels,)
        ROI mask (flattened).
    pixels : int
        Image side length.
    n_cols : int
        Number of columns in the grid.
    cmap : str
        Colormap.
    clip : tuple or "sym" or None
        Clipping mode.
    figsize_per_panel : float
        Size multiplier per panel.
    """
    W_outer = np.asarray(W_outer, dtype=float)
    n_folds = W_outer.shape[0]
    n_rows = int(np.ceil(n_folds / n_cols))

    fig, axes = plt.subplots(n_rows,n_cols,figsize=(figsize_per_panel * n_cols, figsize_per_panel * n_rows),squeeze=False)  

    for k in range(n_folds):
        ax = axes[k // n_cols, k % n_cols]
        img = weights_to_roi_image(W_outer[k], roi_mask_flat, pixels=pixels)
        draw_weight_map(ax,img,cmap=cmap,clip=clip,title=f"Fold {k}",show_colorbar=True)

    for ax in axes.flat[n_folds:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def compute_weight_stats_across_folds(W_outer, eps= 1e-12) -> dict:
    """
    Compute 5 fold-aggregate weight summaries in feature space.

    Returns dict with 1D arrays of shape (n_features,):
    - mean
    - std
    - abs_mean
    - sign_mean
    - Signed max-normalization = w / max(w)
    """
    W_outer = np.asarray(W_outer, dtype=float)
    if W_outer.ndim != 2:
        raise ValueError(f"W_outer must be 2D (n_folds, n_features), got {W_outer.shape}")
    ##### mean and std
    mean = W_outer.mean(axis=0)
    if W_outer.shape[0] > 1:
        std = W_outer.std(axis=0, ddof=1)
    else:
        std = np.full_like(mean, np.nan)
    ##### abs and sign
    abs_mean = np.abs(mean)
    sign_mean = np.sign(mean)

    # Signed max-normalizationSigned max-normalization
    mean_norm = mean / np.nanmax(np.abs(mean))

    return {
        "mean": mean,
        "std": std,
        "abs_mean": abs_mean,
        "sign_mean": sign_mean,
        "mean_norm": mean_norm,
    }


def plot_weight_stat_maps(W_outer,roi_mask_flat, pixels= 100,cmap_mean= ourCmap,
                        cmap_std= "bwr", cmap_abs= ourCmap, cmap_sign= ourCmap,cmap_mean_norm= ourCmap,
                        clip_mean= (-0.0002, 0.0002),
                        clip_std= "sym",
                        clip_abs= (0, 0.0002),
                        clip_sign= (-1, 1),
                        clip_mean_norm= (-1, 1),
                        figsize: tuple = (18, 4),
                        show_colorbar: bool = True,
                        eps: float = 1e-12):
    """
    Compute and plot the 5 maps: mean, std, abs(mean), sign(mean), z-mean.

    Uses:
    - weights_to_roi_image(...)
    - draw_weight_map(...)
    """
    stats = compute_weight_stats_across_folds(W_outer, eps=eps)

    items = [("mean",      stats["mean"],      cmap_mean, clip_mean),
            ("std",       stats["std"],       cmap_std,  clip_std),
            ("abs(mean)", stats["abs_mean"],  cmap_abs,  clip_abs),
            ("sign(mean)",stats["sign_mean"], cmap_sign, clip_sign),
            ("mean_norm", stats["mean_norm"], cmap_mean_norm, clip_mean_norm)    ]

    fig, axes = plt.subplots(1, 5, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, (title, vec, cmap, clip) in zip(axes, items):
        img = weights_to_roi_image(vec, roi_mask_flat, pixels=pixels)
        draw_weight_map(ax, img, cmap=cmap, clip=clip, title=title, show_colorbar=show_colorbar)

    plt.tight_layout()
    plt.show()

    return stats  



def extract_extreme_weight_masks(w_mean,roi_mask_flat, pixels= 100,frac = 0.20):
    """
    Extract top positive and bottom negative weight masks from a mean weight vector.

    Parameters
    ----------
    w_mean : np.ndarray, shape (n_roi_pixels,)
        Mean weight vector in ROI feature space.
    roi_mask_flat : np.ndarray, shape (pixels*pixels,)
        Boolean ROI mask (flattened full image).
    pixels : int
        Image side length (100 -> 10000 pixels).
    frac : float
        Fraction to select from each tail (e.g., 0.20 = 20%).

    Returns
    -------
    pos_mask_flat : np.ndarray, shape (pixels*pixels,), dtype=bool
        Mask of top positive weights.
    neg_mask_flat : np.ndarray, shape (pixels*pixels,), dtype=bool
        Mask of bottom negative weights.
    """
    w_mean = np.asarray(w_mean, dtype=float).ravel()
    roi_mask_flat = np.asarray(roi_mask_flat, dtype=bool).ravel()

    if roi_mask_flat.sum() != w_mean.size:
        raise ValueError(
            f"ROI has {roi_mask_flat.sum()} pixels but w_mean has {w_mean.size} weights"
        )

    # Separate positive and negative weights
    pos_vals = w_mean[w_mean > 0]
    neg_vals = w_mean[w_mean < 0]

    # Initialize ROI-level masks
    pos_roi_mask = np.zeros_like(w_mean, dtype=bool)
    neg_roi_mask = np.zeros_like(w_mean, dtype=bool)

    # Positive threshold (top frac)
    if pos_vals.size > 0:
        pos_thresh = np.percentile(pos_vals, 100 * (1 - frac))
        pos_roi_mask = w_mean >= pos_thresh

    # Negative threshold (bottom frac)
    if neg_vals.size > 0:
        neg_thresh = np.percentile(neg_vals, 100 * frac)
        neg_roi_mask = w_mean <= neg_thresh

    # Map back to full image space
    pos_mask_flat = np.zeros(pixels * pixels, dtype=bool)
    neg_mask_flat = np.zeros(pixels * pixels, dtype=bool)

    pos_mask_flat[roi_mask_flat] = pos_roi_mask
    neg_mask_flat[roi_mask_flat] = neg_roi_mask

    return pos_mask_flat, neg_mask_flat



def average_activation_by_weight_sign(X_roi,roi_mask_flat,
                                    pos_mask_flat,
                                    neg_mask_flat):
    """
    Compute average activation over frames for positive- and negative-weight pixels.

    Parameters
    ----------
    X_roi : np.ndarray, shape (n_roi_pixels, n_frames, n_trials)
        Data restricted to ROI pixels only.
    roi_mask_flat : np.ndarray, shape (pixels*pixels,)
        ROI mask in full image space.
    pos_mask_flat : np.ndarray, shape (pixels*pixels,)
        Positive-weight mask (full image space).
    neg_mask_flat : np.ndarray, shape (pixels*pixels,)
        Negative-weight mask (full image space).

    Returns
    -------
    pos_tc : np.ndarray, shape (n_frames,)
        Mean activation across positive pixels and trials.
    neg_tc : np.ndarray, shape (n_frames,)
        Mean activation across negative pixels and trials.
    """
    X_roi = np.asarray(X_roi, dtype=float)
    roi_mask_flat = np.asarray(roi_mask_flat, dtype=bool)
    pos_mask_flat = np.asarray(pos_mask_flat, dtype=bool)
    neg_mask_flat = np.asarray(neg_mask_flat, dtype=bool)

    n_roi_pixels, n_frames, n_trials = X_roi.shape

    if roi_mask_flat.sum() != n_roi_pixels:
        raise ValueError("roi_mask_flat does not match X_roi pixel dimension")

    # Map full-space masks to ROI index space
    roi_indices = np.where(roi_mask_flat)[0]

    pos_idx = np.isin(roi_indices, np.where(pos_mask_flat)[0])
    neg_idx = np.isin(roi_indices, np.where(neg_mask_flat)[0])

    if not np.any(pos_idx):
        raise ValueError("No positive-weight pixels found in ROI")
    if not np.any(neg_idx):
        raise ValueError("No negative-weight pixels found in ROI")

    # Select pixels
    X_pos = X_roi[pos_idx, :, :]   # (n_pos_pixels, n_frames, n_trials)
    X_neg = X_roi[neg_idx, :, :]   # (n_neg_pixels, n_frames, n_trials)

    # Average across pixels and trials → keep frames
    pos_tc = X_pos.mean(axis=(0, 2))   # (n_frames,)
    neg_tc = X_neg.mean(axis=(0, 2))   # (n_frames,)

    return pos_tc, neg_tc


def plot_pos_neg_timecourses(pos_tc, neg_tc, frame_times= None,
                            title: str = "Positive vs Negative weight pixels"):
    """
    Plot two average timecourses (positive-weight vs negative-weight pixels).

    pos_tc, neg_tc: shape (n_frames,)
    frame_times: optional x-axis values (e.g., ms). If None, uses frame index.
    """
    pos_tc = np.asarray(pos_tc, dtype=float).ravel()
    neg_tc = np.asarray(neg_tc, dtype=float).ravel()
    if pos_tc.size != neg_tc.size:
        raise ValueError(f"pos_tc and neg_tc must have same length, got {pos_tc.size} vs {neg_tc.size}")

    x = np.arange(pos_tc.size) if frame_times is None else np.asarray(frame_times).ravel()
    if x.size != pos_tc.size:
        raise ValueError(f"frame_times must match n_frames, got {x.size} vs {pos_tc.size}")

    plt.figure(figsize=(6, 4))
    plt.plot(x, pos_tc, label="Positive weights", color="magenta")
    plt.plot(x, neg_tc, label="Negative weights", color="green")
    plt.axhline(0, linewidth=1)
    plt.xlabel("Frame" if frame_times is None else "Time")
    plt.ylabel("Mean activation")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
