"""
Method B, activation-based variant — single session: 110209_a_1,5

Instead of correlating the SVM's oof_trial_score (decision output) between
V1 and V2, this correlates a new signal built directly from the raw VSD
pixel activation, restricted to each window's most classifier-informative
pixels:

For each window w (with center frame c), for each area (V1, V2) separately:
    1. Take that window's own model's weight vector -- x_mean_window[w, :],
       shape (n_roi_pixels,). This is ALREADY restricted to this area's own
       ROI (e.g. 4097 pixels out of the full 10000-pixel grid), not the
       full grid.
    2. Select the top 20% of ROI-columns by |weight| -> a window-specific
       mask, in ROI-column-space.
    3. Translate that mask to raw-pixel indices (0-9999) using this area's
       own roi_mask_flat (boolean, length 10000): column i of
       x_mean_window corresponds to raw pixel np.where(roi_mask_flat)[0][i].
    4. On the z-scored raw VSD data (full 10000-pixel grid), average
       activation over:
         - just those translated raw pixels
         - the 5 raw frames belonging to this window (c-2 .. c+2)
       separately for every trial.
    5. This gives one "activation" value per trial per window -- this
       REPLACES oof_trial_score as the signal fed into Method B.

Both V1 and V2 use masks drawn from their OWN model's weights, but applied
to the SAME underlying z-scored VSD array (one shared recording, per user
confirmation) -- V1's high-|weight| pixels sit in V1 cortex, V2's in V2
cortex, so no separate anatomical ROI masking is needed on top.

Z-scoring: per-trial baseline-mean subtraction, then a single POOLED std
(pooled across all trials, both classes, and baseline frames together) --
via the user-provided zscore_dataset_pixelwise_trials function, exactly as
used for the original classifier's own z-scoring. baseline_frames=(1,25).

Frame indexing: per user confirmation, the model's frame/center numbering
(e.g. centers from Weights_acrossT / config) indexes directly into the raw
VSD array -- no offset conversion needed.

Once v1_activation / v2_activation (n_windows x n_trials) are built, they
are fed into the SAME Method B lag-curve engine used for oof_trial_score.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ZERO_FRAME = 27
FRAME_DURATION_MS = 10

PHASE1_T_START = 0
PHASE1_T_STOP = 120

PHASE2_T_START = 120
PHASE2_T_STOP = 250

LAG_MIN = -50
LAG_MAX = 50
LAG_STEP = 10

TOP_PCT = 0.20  # top 20% most extreme |weight| pixels
BASELINE_FRAMES = (1, 25)  # per user's zscore function convention

RESULTS_V1_DIR = Path(r"C:\project\vsdi-face-decoding\results\V1")
RESULTS_V2_DIR = Path(r"C:\project\vsdi-face-decoding\results\V2")
RAW_DATA_DIR = Path(r"C:\project\vsdi-face-decoding\data\processed\condsXn")

PAIR_KEY = "110209_a_1,5"
V1_FOLDER_HINT = "110209a15_V1"
V2_FOLDER_HINT = "110209a15_V2"
FACE_FILE = "condsXn1_110209a.npy"     # condition 1 = face
NONFACE_FILE = "condsXn5_110209a.npy"  # condition 5 = nonface


# ---------------------------------------------------------------------------
# User-provided functions (verbatim)
# ---------------------------------------------------------------------------
def build_X_y(face_file, nonface_file, data_dir):
    """
    Load one face condition and one non-face condition
    and return X (all trials) and y (labels).

    X shape: (pixels, frames, trials)
    y shape: (trials,)
    """
    data_dir = Path(data_dir)
    X_face = np.load(data_dir / face_file)
    X_non = np.load(data_dir / nonface_file)
    print("Loaded shapes:")
    print(" face    :", X_face.shape)
    print(" nonface :", X_non.shape)
    n = min(X_face.shape[2], X_non.shape[2])
    X_face = X_face[:, :, :n]
    X_non = X_non[:, :, :n]
    X = np.concatenate([X_face, X_non], axis=2)
    FACE_LABEL = 1
    NONFACE_LABEL = 0
    y = np.array([FACE_LABEL] * n + [NONFACE_LABEL] * n)
    print("Final:")
    print(" X shape:", X.shape)
    print(" y:", y)
    return X, y, {"n_trials_per_class": n,
                  "face_file": face_file,
                  "nonface_file": nonface_file}


def zscore_dataset_pixelwise_trials(X, baseline_frames=(0, 24), eps: float = 1e-8, ddof: int = 0):
    """
    Updated Pixel-wise Z-score:
    1. Subtract trial-specific baseline mean from each trial.
    2. Calculate STD of those 'mean-centered' baseline segments pooled across trials.
    3. Divide by that pooled STD.
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


# ---------------------------------------------------------------------------
# Loading model results (weights, centers)
# ---------------------------------------------------------------------------
def load_experiment(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    d = np.load(run_dir / "result.npz", allow_pickle=True)
    results = d["results"].item()
    d_roi = np.load(run_dir / "ROI_mask_path.npz", allow_pickle=True)
    roi_mask_path = str(d_roi["roi_mask_path"])  # path to a .npy file
    roi_mask_flat = np.load(roi_mask_path).astype(bool).ravel()  # (10000,) boolean
    return config, results, roi_mask_flat


def find_folder(area_dir, hint):
    matches = [p for p in area_dir.iterdir() if p.is_dir() and hint in p.name]
    if len(matches) == 0:
        raise FileNotFoundError(f"No folder in {area_dir} containing '{hint}'")
    if len(matches) > 1:
        matches.sort()
        return matches[-1]
    return matches[0]


def centers_to_time_ms(centers):
    return (np.asarray(centers) - ZERO_FRAME) * FRAME_DURATION_MS


def get_phase_indices(time_ms, t_start, t_stop):
    mask = (time_ms >= t_start) & (time_ms <= t_stop)
    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Extreme-weight-mask activation signal
# ---------------------------------------------------------------------------
def weights_to_roi_image(w_roi, roi_mask_flat, pixels=100, fill_value=np.nan):
    """
    Map a 1D weight vector defined on ROI pixels back into full image space.
    (verbatim from user)
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


def plot_extreme_weights_across_time(x_mean_window, roi_mask_flat, centers, time_ms,
                                       area_name, pair_key, top_pct=TOP_PCT,
                                       n_snapshots=12, pixels=100):
    """
    mimg-style grid: for a set of evenly-spaced windows across the full time
    range, show ONLY the top-pct extreme-|weight| pixels' actual weight
    values (everything else left blank/NaN) -- one image per snapshot.
    """
    n_windows = x_mean_window.shape[0]
    snapshot_idx = np.linspace(0, n_windows - 1, n_snapshots).astype(int)
    snapshot_idx = np.unique(snapshot_idx)
    n_shown = len(snapshot_idx)

    ncols = 4
    nrows = int(np.ceil(n_shown / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # shared color scale across all snapshots, using only the extreme values
    # that will actually be shown (computed below) so panels are comparable
    extreme_imgs = []
    for w in snapshot_idx:
        mask = top_pct_mask(x_mean_window[w, :], pct=top_pct)
        w_extreme_only = np.where(mask, x_mean_window[w, :], np.nan)
        img = weights_to_roi_image(w_extreme_only, roi_mask_flat, pixels=pixels, fill_value=np.nan)
        extreme_imgs.append(img)
    vmax = np.nanmax([np.nanmax(np.abs(img)) for img in extreme_imgs])

    for i, w in enumerate(snapshot_idx):
        ax = axes[i]
        im = ax.imshow(extreme_imgs[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"t={time_ms[w]:.0f}ms", fontsize=9)
        ax.axis("off")

    for j in range(n_shown, len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="weight (extreme pixels only)")

    fig.suptitle(f"{area_name} top-{int(top_pct*100)}% extreme-weight pixels across time — {pair_key}")
    return fig
    """Boolean mask (in ROI-column-space) selecting the top pct of columns
    by |weight|."""
    n_roi_pixels = len(weight_vector)
    n_select = int(round(n_roi_pixels * pct))
    abs_w = np.abs(weight_vector)
    top_idx = np.argpartition(abs_w, -n_select)[-n_select:]
    mask = np.zeros(n_roi_pixels, dtype=bool)
    mask[top_idx] = True
    return mask


def top_pct_mask(weight_vector, pct=TOP_PCT):
    """Boolean mask (in ROI-column-space) selecting the top pct of columns
    by |weight|."""
    n_roi_pixels = len(weight_vector)
    n_select = int(round(n_roi_pixels * pct))
    abs_w = np.abs(weight_vector)
    top_idx = np.argpartition(abs_w, -n_select)[-n_select:]
    mask = np.zeros(n_roi_pixels, dtype=bool)
    mask[top_idx] = True
    return mask


def build_activation_signal(X_z, x_mean_window, roi_mask_flat, centers, top_pct=TOP_PCT):
    """
    For each window (row of x_mean_window, shape (n_windows, n_roi_pixels),
    with matching center in centers), build the top-pct-|weight| mask IN
    ROI-COLUMN-SPACE, translate that mask to raw-pixel indices (0-9999)
    via roi_mask_flat, then average the z-scored activation over:
        - those raw pixels
        - the window's 5 raw frames (c-2..c+2)
    per trial.

    X_z: (10000, frames, trials) z-scored raw VSD data (full pixel grid)
    x_mean_window: (n_windows, n_roi_pixels) -- this area's own model weights,
        restricted to this area's own ROI pixels
    roi_mask_flat: (10000,) boolean, True at this area's ROI pixel positions.
        Column i of x_mean_window corresponds to raw pixel
        np.where(roi_mask_flat)[0][i].
    centers: (n_windows,) frame-center per window, directly indexes X_z

    Returns: activation (n_windows, n_trials)
    """
    n_windows, n_roi_pixels = x_mean_window.shape
    assert len(centers) == n_windows, "centers/x_mean_window window count mismatch"

    roi_pixel_indices = np.where(roi_mask_flat)[0]  # (n_roi_pixels,) -> raw pixel index
    assert len(roi_pixel_indices) == n_roi_pixels, (
        f"ROI mask has {len(roi_pixel_indices)} True pixels but x_mean_window has "
        f"{n_roi_pixels} columns -- mismatch, check roi_mask_flat/x_mean_window alignment"
    )

    n_pixels_full, n_frames, n_trials = X_z.shape
    activation = np.full((n_windows, n_trials), np.nan)
    dropped = []

    for w in range(n_windows):
        c = int(centers[w])
        f_start, f_end = c - 2, c + 2  # inclusive 5-frame window
        if f_start < 0 or f_end >= n_frames:
            dropped.append((w, c))
            continue
        roi_mask_this_window = top_pct_mask(x_mean_window[w, :], pct=top_pct)  # (n_roi_pixels,)
        raw_pixel_idx = roi_pixel_indices[roi_mask_this_window]  # translate to raw-pixel indices
        chunk = X_z[raw_pixel_idx, f_start:f_end + 1, :]  # (n_selected_pixels, 5, n_trials)
        activation[w, :] = chunk.mean(axis=(0, 1))  # mean over pixels AND frames, per trial

    if dropped:
        print(f"  WARNING: {len(dropped)} windows dropped (frame range out of bounds): "
              f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}")

    return activation


# ---------------------------------------------------------------------------
# Method B core (identical engine, reused on the new activation signal)
# ---------------------------------------------------------------------------
def compute_lag_curve(v1_signal, v2_signal, n_windows, ref_idx, lag_min, lag_max, lag_step):
    index_step_ms = FRAME_DURATION_MS
    n_trials = v2_signal.shape[1]
    assert v1_signal.shape[1] == n_trials, "V1/V2 trial count mismatch!"

    lags = np.arange(lag_min, lag_max + lag_step, lag_step)
    mean_r_list, surviving_lags = [], []
    v1_idx_by_lag = {}

    for lag in lags:
        shift = int(round(lag / index_step_ms))
        v1_idx = ref_idx + shift
        if v1_idx.min() < 0 or v1_idx.max() > n_windows - 1:
            continue
        v2_vals = v2_signal[ref_idx, :]
        v1_vals = v1_signal[v1_idx, :]
        trial_rs = np.full(n_trials, np.nan)
        for tr in range(n_trials):
            v2_t, v1_t = v2_vals[:, tr], v1_vals[:, tr]
            if np.std(v2_t) == 0 or np.std(v1_t) == 0 or np.any(np.isnan(v2_t)) or np.any(np.isnan(v1_t)):
                continue
            r, _ = pearsonr(v2_t, v1_t)
            trial_rs[tr] = r
        mean_r_list.append(np.nanmean(trial_rs))
        surviving_lags.append(lag)
        v1_idx_by_lag[lag] = v1_idx

    return np.array(surviving_lags), np.array(mean_r_list), v1_idx_by_lag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. raw VSD data, shared by both areas
    X, y, meta = build_X_y(FACE_FILE, NONFACE_FILE, RAW_DATA_DIR)
    true_label = np.where(y == 1, 1, -1)  # convert {1,0} -> {+1,-1} for consistency

    # 2. z-score once
    X_z, mean_per_trial, std_pooled = zscore_dataset_pixelwise_trials(X, baseline_frames=BASELINE_FRAMES)
    print(f"X_z shape: {X_z.shape}, std_pooled shape: {std_pooled.shape}")

    # 3. load V1 / V2 models
    v1_folder = find_folder(RESULTS_V1_DIR, V1_FOLDER_HINT)
    v2_folder = find_folder(RESULTS_V2_DIR, V2_FOLDER_HINT)
    v1_config, v1_results, v1_roi_mask = load_experiment(v1_folder)
    v2_config, v2_results, v2_roi_mask = load_experiment(v2_folder)

    v1_weights = v1_results["w_mean_windows"]  # (n_windows, n_roi_pixels)
    v2_weights = v2_results["w_mean_windows"]
    v1_centers = v1_results["centers"]
    v2_centers = v2_results["centers"]
    print(f"V1 weights shape: {v1_weights.shape}, V2 weights shape: {v2_weights.shape}")
    print(f"V1 ROI pixels: {v1_roi_mask.sum()}, V2 ROI pixels: {v2_roi_mask.sum()}")

    assert np.array_equal(v1_centers, v2_centers), "V1/V2 centers differ -- stopping"
    centers = v1_centers
    n_windows = len(centers)

    # sanity check: trial count in raw data must match model's trial count
    n_trials_raw = X_z.shape[2]
    n_trials_model = v1_results["oof_trial_score"].shape[1]
    if n_trials_raw != n_trials_model:
        print(f"  WARNING: raw data has {n_trials_raw} trials, model has {n_trials_model} trials -- mismatch!")

    # 4-5. build activation signals
    print("\nBuilding V1 activation signal...")
    v1_activation = build_activation_signal(X_z, v1_weights, v1_roi_mask, centers, top_pct=TOP_PCT)
    print("Building V2 activation signal...")
    v2_activation = build_activation_signal(X_z, v2_weights, v2_roi_mask, centers, top_pct=TOP_PCT)

    time_ms = centers_to_time_ms(centers)
    print(f"\nFull time range: {time_ms.min():.0f} to {time_ms.max():.0f} ms ({n_windows} windows)")

    # -----------------------------------------------------------------
    # Figure: extreme-weight maps across time, mimg-style, V1 and V2
    # -----------------------------------------------------------------
    plot_extreme_weights_across_time(v1_weights, v1_roi_mask, centers, time_ms,
                                      area_name="V1", pair_key=PAIR_KEY, top_pct=TOP_PCT)
    plot_extreme_weights_across_time(v2_weights, v2_roi_mask, centers, time_ms,
                                      area_name="V2", pair_key=PAIR_KEY, top_pct=TOP_PCT)

    # -----------------------------------------------------------------
    # Figure: grand-average V1/V2 activation across time (sanity check)
    # -----------------------------------------------------------------
    fig0, ax0 = plt.subplots(figsize=(9, 4))
    ax0.plot(time_ms, np.nanmean(v1_activation, axis=1), label="V1 (grand avg activation)", color="tab:blue")
    ax0.plot(time_ms, np.nanmean(v2_activation, axis=1), label="V2 (grand avg activation)", color="tab:orange")
    ax0.axvspan(PHASE1_T_START, PHASE1_T_STOP, color="gray", alpha=0.15, label="phase 1 window")
    ax0.axhline(0, color="black", linewidth=0.5)
    ax0.set_xlabel("time (ms)")
    ax0.set_ylabel("mean z-scored activation\n(top-20%-|weight| pixels)")
    ax0.set_title(f"Grand-average V1/V2 extreme-weight-pixel activation — {PAIR_KEY}")
    ax0.legend()
    fig0.tight_layout()

    # 6. Method B on the new signal -- run for phase 1 and phase 2 separately.
    # NOTE: v1_activation / v2_activation already cover ALL windows (full
    # time range), so no need to rebuild them per phase -- only ref_idx
    # (which windows count as "reference" points) differs between phases.
    phases = [
        ("phase 1", PHASE1_T_START, PHASE1_T_STOP),
        ("phase 2", PHASE2_T_START, PHASE2_T_STOP),
    ]
    phase_results = {}
    for phase_name, t_start, t_stop in phases:
        ref_idx_phase = get_phase_indices(time_ms, t_start, t_stop)
        ref_times_phase = time_ms[ref_idx_phase]
        print(f"\n{phase_name} ({t_start}-{t_stop}ms): "
              f"{len(ref_idx_phase)} reference timepoints: {ref_times_phase}")

        lags, mean_r, v1_idx_by_lag = compute_lag_curve(
            v1_activation, v2_activation, n_windows, ref_idx_phase, LAG_MIN, LAG_MAX, LAG_STEP,
        )
        best_lag = lags[np.nanargmax(mean_r)]
        print(f"{phase_name}: best lag = {best_lag:+d} ms "
              f"({'V2 leads V1' if best_lag > 0 else 'V1 leads V2' if best_lag < 0 else 'no lead'})")
        for lag, r in zip(lags, mean_r):
            marker = "  <-- max" if r == np.nanmax(mean_r) else ""
            print(f"  {lag:>+4d}ms: r={r:.4f}{marker}")

        phase_results[phase_name] = {
            "t_start": t_start, "t_stop": t_stop,
            "lags": lags, "mean_r": mean_r, "best_lag": best_lag,
        }

    # -----------------------------------------------------------------
    # Figure: phase 1 and phase 2 lag curves as subplots, same figure
    # -----------------------------------------------------------------
    fig_phases, axes_phases = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (phase_name, t_start, t_stop) in zip(axes_phases, phases):
        res = phase_results[phase_name]
        ax.plot(res["lags"], res["mean_r"], marker="o")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(res["best_lag"], color="red", linestyle=":", linewidth=1.5,
                   label=f"best lag = {res['best_lag']:+d} ms")
        ax.set_xlabel("lag (ms)  [positive = V2 leads V1]")
        ax.set_title(f"{phase_name} ({t_start}-{t_stop}ms)")
        ax.legend()
    axes_phases[0].set_ylabel("mean per-trial correlation (r)")
    fig_phases.suptitle(f"Method B (extreme-weight activation) — {PAIR_KEY}")
    fig_phases.tight_layout()

    plt.show()