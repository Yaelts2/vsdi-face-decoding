"""
explain_pipeline_one_session.py

Run ONCE on a single (date, session, conditions) pair to see every stage
of the V1-V2 lag-estimation pipeline with your own eyes, using REAL data:

  1. Raw trial-averaged V1/V2 curves (context only -- not what the lag
     search actually uses)
  2. Example single-trial scatter plots at one timepoint, a few candidate
     lags -- this is literally the raw material behind ONE cell of the
     correlation grid: every dot is one trial
  3. The full correlation grid: every timepoint x every candidate lag
     (heatmap) -- one cell = one across-trial correlation
  4. That grid averaged across timepoints -> the final correlation-vs-lag
     curve, with the winning lag marked

Edit PAIR_KEY below to choose which of your 16 sessions to inspect (or
leave as None to use the first one found).
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# make sure Python can find v1_v2_lag_crosscorr.py regardless of the
# current working directory -- edit this if that file lives somewhere
# other than the same folder as this script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v1_v2_lag_crosscorr import (
    ZERO_FRAME, FRAME_DURATION_MS, WINDOW_MS, LAG_RANGE_MS, LAG_STEP_MS,
    load_experiment, find_v1_v2_pairs,
)

RESULTS_ROOT = r"C:\project\vsdi-face-decoding\results"  # <-- edit if needed
PAIR_KEY = "110209_a_2,4"  # <-- e.g. "030209_a_1,5" ; leave None to use the first pair found
OUT_DIR = Path(r"C:\project\vsdi-face-decoding\scripts\v1_v2_lag_crosscorr_analysis\pipeline_explainer_plots")


def plot_raw_curves(pair_key, time_ms, v1_scores, v2_scores, out_dir):
    v1_mean = np.nanmean(v1_scores, axis=1)
    v2_mean = np.nanmean(v2_scores, axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(time_ms, v2_mean, color="#2a78d6", label="V2 (trial-avg score)")
    ax.plot(time_ms, v1_mean, color="#eb6834", linestyle="--", label="V1 (trial-avg score)")
    ax.axvspan(*WINDOW_MS, color="gray", alpha=0.15, label="Analysis window")
    ax.axvline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Mean decision score (a.u.)")
    ax.set_title(f"{pair_key}: step 1 -- raw trial-averaged curves (context only)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "1_raw_curves.png", dpi=120)
    plt.close(fig)


def plot_example_scatters(pair_key, time_ms, v1_scores, v2_scores, t_idx, lags_ms, out_dir):
    fig, axes = plt.subplots(1, len(lags_ms), figsize=(4.3 * len(lags_ms), 4))
    v2_t = v2_scores[t_idx, :]
    t_ms = time_ms[t_idx]

    for ax, lag_ms in zip(axes, lags_ms):
        lag_frames = int(round(lag_ms / FRAME_DURATION_MS))
        v1_idx = t_idx + lag_frames
        if v1_idx < 0 or v1_idx >= v1_scores.shape[0]:
            ax.set_title(f"lag={lag_ms:+d}ms (out of range)")
            continue
        v1_t = v1_scores[v1_idx, :]
        r = np.corrcoef(v2_t, v1_t)[0, 1]
        ax.scatter(v2_t, v1_t, color="#4a3aa7", alpha=0.7, s=25)
        ax.set_xlabel(f"V2 score @ {t_ms:.0f}ms")
        ax.set_ylabel(f"V1 score @ {t_ms+lag_ms:.0f}ms")
        ax.set_title(f"lag = {lag_ms:+d}ms\nr = {r:.2f}  (n={len(v2_t)} trials)")

    fig.suptitle(f"{pair_key}: step 2 -- one timepoint, candidate lags\n"
                 "each dot = ONE trial; this scatter is what ONE correlation number is built from")
    plt.tight_layout()
    fig.savefig(out_dir / "2_example_scatters.png", dpi=120)
    plt.close(fig)


def build_correlation_grid(v1_scores, v2_scores, window_idx, lags_ms):
    n_win = len(window_idx)
    grid = np.full((len(lags_ms), n_win), np.nan)  # rows=lag, cols=timepoint
    for li, lag_ms in enumerate(lags_ms):
        lag_frames = int(round(lag_ms / FRAME_DURATION_MS))
        for ti, t_idx in enumerate(window_idx):
            v1_idx = t_idx + lag_frames
            if v1_idx < 0 or v1_idx >= v1_scores.shape[0]:
                continue
            r = np.corrcoef(v2_scores[t_idx, :], v1_scores[v1_idx, :])[0, 1]
            grid[li, ti] = np.clip(r, -0.999999, 0.999999)
    avg_curve = np.tanh(np.nanmean(np.arctanh(grid), axis=1))  # avg across timepoints
    return grid, avg_curve


def plot_grid_and_average(pair_key, window_time_ms, lags_ms, grid, avg_curve, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                    extent=[window_time_ms[0], window_time_ms[-1], lags_ms[0], lags_ms[-1]])
    ax.set_xlabel("Timepoint (ms)")
    ax.set_ylabel("Candidate lag (ms)")
    ax.set_title(f"{pair_key}: step 3 -- full correlation grid\n"
                 "(one cell = one correlation-across-trials number, at this timepoint & lag)")
    fig.colorbar(im, ax=ax, label="correlation (r)")

    ax2 = axes[1]
    ax2.plot(lags_ms, avg_curve, color="#199e70")
    best_i = np.nanargmax(avg_curve)
    ax2.axvline(lags_ms[best_i], color="red", linestyle="--", label=f"lag* = {lags_ms[best_i]:+.0f} ms")
    ax2.axvline(0, color="k", linestyle=":", linewidth=1)
    ax2.set_xlabel("Candidate lag (ms)  [positive = V2 leads V1]")
    ax2.set_ylabel("Avg correlation (r)")
    ax2.set_title("step 4 -- grid averaged across timepoints (each row -> one point)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "3_grid_and_average.png", dpi=120)
    plt.close(fig)


def plot_lag_overlay_grid(pair_key, time_ms, v1_scores, v2_scores, window_idx, lags_ms, avg_curve, out_dir):
    """
    One small panel per candidate lag: V2's trial-averaged curve (solid
    blue) vs. V1's trial-averaged curve SHIFTED by that lag (dashed
    orange), both plotted on the same time axis so you can see directly
    whether shifting V1 by this lag pulls it into alignment with V2.
    The winning lag's panel is outlined in red.

    NOTE: this uses trial-AVERAGED curves purely so the alignment is
    visible to the eye. The actual lag search (and the r printed in each
    panel's title) is computed from per-trial scores, not from this
    averaged curve -- this plot is a visual companion to that number,
    not a recomputation of it.
    """
    v1_mean_full = np.nanmean(v1_scores, axis=1)
    v2_mean_full = np.nanmean(v2_scores, axis=1)
    v2_window = v2_mean_full[window_idx]
    window_time = time_ms[window_idx]

    n_lags = len(lags_ms)
    ncols = 6
    nrows = int(np.ceil(n_lags / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    best_i = int(np.nanargmax(avg_curve))

    for i, lag_ms in enumerate(lags_ms):
        ax = axes[i]
        lag_frames = int(round(lag_ms / FRAME_DURATION_MS))
        v1_shifted = np.full_like(v2_window, np.nan)
        for j, t_idx in enumerate(window_idx):
            v1_idx = t_idx + lag_frames
            if 0 <= v1_idx < len(v1_mean_full):
                v1_shifted[j] = v1_mean_full[v1_idx]

        ax.plot(window_time, v2_window, color="#2a78d6", linewidth=1.3, label="V2")
        ax.plot(window_time, v1_shifted, color="#eb6834", linewidth=1.3, linestyle="--", label="V1 shifted")
        r = avg_curve[i]
        ax.set_title(f"lag={lag_ms:+d}ms  r={r:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
        if i == best_i:
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)

    for k in range(n_lags, len(axes)):
        axes[k].axis("off")

    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"{pair_key}: V2 (solid) vs. V1 shifted by each candidate lag (dashed)\n"
                 "red border = winning lag -- this is the visual counterpart to the correlation curve",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "4_lag_overlay_grid.png", dpi=110)
    plt.close(fig)


def main():
    pairs = find_v1_v2_pairs(RESULTS_ROOT)
    pair_key = PAIR_KEY or sorted(pairs.keys())[0]
    print(f"Using pair: {pair_key}")
    d = pairs[pair_key]

    _, v1_results, _ = load_experiment(d["V1"])
    _, v2_results, _ = load_experiment(d["V2"])
    v1_scores = v1_results["oof_trial_score"]
    v2_scores = v2_results["oof_trial_score"]
    centers = np.asarray(v1_results["centers"])
    time_ms = (centers - ZERO_FRAME) * FRAME_DURATION_MS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_raw_curves(pair_key, time_ms, v1_scores, v2_scores, OUT_DIR)

    window_mask = (time_ms >= WINDOW_MS[0]) & (time_ms <= WINDOW_MS[1])
    window_idx = np.where(window_mask)[0]
    example_t_idx = window_idx[len(window_idx) // 2]  # a mid-window timepoint
    example_lags_ms = [-50, 0, 40]
    plot_example_scatters(pair_key, time_ms, v1_scores, v2_scores,
                           example_t_idx, example_lags_ms, OUT_DIR)

    lags_ms = np.arange(-LAG_RANGE_MS, LAG_RANGE_MS + 1, LAG_STEP_MS)
    grid, avg_curve = build_correlation_grid(v1_scores, v2_scores, window_idx, lags_ms)
    plot_grid_and_average(pair_key, time_ms[window_idx], lags_ms, grid, avg_curve, OUT_DIR)
    plot_lag_overlay_grid(pair_key, time_ms, v1_scores, v2_scores, window_idx, lags_ms, avg_curve, OUT_DIR)

    print(f"Saved 4 plots to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()