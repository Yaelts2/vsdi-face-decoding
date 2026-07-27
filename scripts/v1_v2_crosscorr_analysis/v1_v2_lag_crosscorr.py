"""
v1_v2_lag_crosscorr.py

Method A: for each (date, session, conditions) pair, estimate the
V1-vs-V2 timing offset in the phase-2 window by finding the lag that
maximizes the across-trial correlation between V2's and V1's
out-of-fold decision-function scores (oof_trial_score), then tests
whether that lag is systematically positive (V2 leads V1) across all
16 pairs using a one-sample Wilcoxon signed-rank test.

SIGN CONVENTION (read carefully -- easy to flip by accident):
  For candidate lag L (ms), at each timepoint t in the analysis window
  we correlate V2_score[t, :] against V1_score[t + L, :], across
  trials. If the best lag L* is POSITIVE, that means V1's evidence at
  a LATER timepoint (t + L*) best matches V2's evidence NOW (t) -- i.e.
  V2's pattern leads V1's by L* ms. This is the hypothesized direction.
  A NEGATIVE L* means V1 leads V2 (feedforward-consistent, opposite of
  the hypothesis).

ANALYSIS WINDOW: fixed at 50-250 ms post-stimulus-onset for every pair
(no per-pair manual picking), per team decision. This may include some
phase-1 tail for sessions where phase 1 resolves late -- if so, that
would bias L* toward negative (working AGAINST the hypothesis, not
inflating it), but worth eyeballing the per-pair correlation-vs-lag
curves before trusting the summary number blindly.

Requires: numpy, scipy, pandas
"""

import re
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ---- fixed constants (confirmed identical across all runs) ----
ZERO_FRAME = 27
FRAME_DURATION_MS = 10

# ---- analysis parameters ----
WINDOW_MS = (50, 250)      # phase-2 analysis window, fixed for all pairs
LAG_RANGE_MS = 150         # search +/- this many ms
LAG_STEP_MS = FRAME_DURATION_MS  # finest resolution the data supports


RUN_NAME_RE = re.compile(
    r"^sliding_window__"
    r"(?P<date>\d{6})"
    r"(?P<session>[a-z])"
    r"(?P<cond1>\d)"
    r"(?P<cond2>\d)"
    r"_V(?P<area>\d)"
    r"_frame(?P<start_frame>\d+)-(?P<stop_frame>\d+)"
    r"__SVM_(?P<n_splits>\d+)foldCV"
    r"_{2,}"
    r"(?P<timestamp>[\d_-]+)$"
)


def load_experiment(run_dir):
    """As provided: loads config.json, result.npz, ROI mask path."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)
    d = np.load(run_dir / "result.npz", allow_pickle=True)
    results = d["results"].item()
    d_roi = np.load(run_dir / "ROI_mask_path.npz", allow_pickle=True)
    ROI_mask_path = str(d_roi["roi_mask_path"])
    return config, results, ROI_mask_path


def compute_avg_corr_curve(v1_scores, v2_scores, window_idx, lags_ms):
    """
    Shared helper: for a given (n_windows x n_trials) V1/V2 score matrix
    pair, return one Fisher-z-averaged correlation value per candidate
    lag. Same math as fisher_z_mean_corr/estimate_pair_lag, exposed here
    so other scripts (e.g. the shuffle control) can reuse it exactly
    rather than re-implementing it and risking drift.
    """
    avg_curve = np.full(len(lags_ms), np.nan)
    for li, lag_ms in enumerate(lags_ms):
        lag_frames = int(round(lag_ms / FRAME_DURATION_MS))
        zs = []
        for t_idx in window_idx:
            v1_idx = t_idx + lag_frames
            if v1_idx < 0 or v1_idx >= v1_scores.shape[0]:
                continue
            r = np.corrcoef(v2_scores[t_idx, :], v1_scores[v1_idx, :])[0, 1]
            r = np.clip(r, -0.999999, 0.999999)
            zs.append(np.arctanh(r))
        if zs:
            avg_curve[li] = np.tanh(np.nanmean(zs))
    return avg_curve


def find_v1_v2_pairs(results_root):
    """
    Scan results_root/V1 and results_root/V2, parse folder names, and
    return a dict: pair_key -> {'V1': run_dir, 'V2': run_dir}.
    Raises if any pair_key is missing a side or duplicated (should not
    happen given the inventory check already run, but re-verified here
    defensively since this script may be run standalone).
    """
    results_root = Path(results_root)
    found = {"V1": {}, "V2": {}}

    for area in ("V1", "V2"):
        area_dir = results_root / area
        for run_dir in sorted(area_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            m = RUN_NAME_RE.match(run_dir.name)
            if m is None:
                raise ValueError(f"Folder does not match expected pattern: {run_dir}")
            d = m.groupdict()
            if f"V{d['area']}" != area:
                raise ValueError(f"Area mismatch for {run_dir}: folder says V{d['area']} but lives under {area}/")
            pair_key = f"{d['date']}_{d['session']}_{d['cond1']},{d['cond2']}"
            if pair_key in found[area]:
                raise ValueError(f"Duplicate {area} run for {pair_key}: {found[area][pair_key]} and {run_dir}")
            found[area][pair_key] = run_dir

    v1_keys, v2_keys = set(found["V1"]), set(found["V2"])
    missing_v2 = v1_keys - v2_keys
    missing_v1 = v2_keys - v1_keys
    if missing_v2:
        raise ValueError(f"Pairs with V1 but no V2: {missing_v2}")
    if missing_v1:
        raise ValueError(f"Pairs with V2 but no V1: {missing_v1}")

    pairs = {k: {"V1": found["V1"][k], "V2": found["V2"][k]} for k in v1_keys}
    return pairs


def fisher_z_mean_corr(v2_scores_window, v1_scores_full, window_idx, lag_frames):
    """
    For one candidate lag, compute the across-trial correlation between
    V2's score and V1's score at each timepoint in the analysis window,
    Fisher-z average across timepoints, and return the back-transformed
    mean correlation (a single number for this lag).

    v2_scores_window : (n_window_timepoints, n_trials) -- V2 scores
                        already restricted to the analysis window
    v1_scores_full    : (n_windows_full, n_trials) -- V1 scores, FULL
                        time range (so we can index t + lag_frames)
    window_idx         : indices into the FULL time axis corresponding
                          to v2_scores_window's rows (same order)
    lag_frames          : integer frame shift to apply to V1

    Returns np.nan if any required V1 index falls outside the array
    (shouldn't happen given the margins chosen, but checked defensively).
    """
    n_win = v2_scores_window.shape[0]
    zs = []
    for i in range(n_win):
        v1_idx = window_idx[i] + lag_frames
        if v1_idx < 0 or v1_idx >= v1_scores_full.shape[0]:
            return np.nan
        v2_t = v2_scores_window[i, :]
        v1_t = v1_scores_full[v1_idx, :]
        r = np.corrcoef(v2_t, v1_t)[0, 1]
        r = np.clip(r, -0.999999, 0.999999)  # avoid inf at exactly +/-1
        zs.append(np.arctanh(r))
    mean_z = np.nanmean(zs)
    return np.tanh(mean_z)


def estimate_pair_lag(v1_run_dir, v2_run_dir):
    """
    Load one V1/V2 pair, compute the correlation-vs-lag curve over the
    fixed analysis window, and return (lag_star_ms, best_corr,
    lags_ms, corrs) -- the last two for optional plotting/inspection.
    """
    _, v1_results, _ = load_experiment(v1_run_dir)
    _, v2_results, _ = load_experiment(v2_run_dir)

    v1_scores = v1_results["oof_trial_score"]  # (n_windows, n_trials)
    v2_scores = v2_results["oof_trial_score"]

    v1_centers = np.asarray(v1_results["centers"])
    v2_centers = np.asarray(v2_results["centers"])
    if not np.array_equal(v1_centers, v2_centers):
        raise ValueError("V1 and V2 have different window centers -- cannot align by index.")
    if v1_scores.shape[1] != v2_scores.shape[1]:
        raise ValueError(
            f"V1 has {v1_scores.shape[1]} trials but V2 has {v2_scores.shape[1]} -- "
            "trial-count mismatch, positional alignment invalid."
        )

    centers = v1_centers
    time_ms = (centers - ZERO_FRAME) * FRAME_DURATION_MS

    step_frames = int(np.round(np.median(np.diff(centers))))  # should be 1
    ms_per_step = step_frames * FRAME_DURATION_MS

    window_mask = (time_ms >= WINDOW_MS[0]) & (time_ms <= WINDOW_MS[1])
    if not np.any(window_mask):
        raise ValueError(f"No windows fall inside {WINDOW_MS} ms for this pair.")
    window_idx = np.where(window_mask)[0]
    v2_scores_window = v2_scores[window_idx, :]

    lags_ms = np.arange(-LAG_RANGE_MS, LAG_RANGE_MS + 1, LAG_STEP_MS)
    corrs = np.full(lags_ms.shape, np.nan)

    for li, lag_ms in enumerate(lags_ms):
        if lag_ms % ms_per_step != 0:
            continue  # lag not reachable at this time resolution
        lag_frames = int(round(lag_ms / ms_per_step)) * step_frames
        corrs[li] = fisher_z_mean_corr(v2_scores_window, v1_scores, window_idx, lag_frames)

    if np.all(np.isnan(corrs)):
        raise ValueError("All lags produced NaN -- check window/lag range vs. available frames.")

    best_i = np.nanargmax(corrs)
    lag_star_ms = lags_ms[best_i]
    best_corr = corrs[best_i]

    # trial-averaged curves, FULL time range -- for the raw-data sanity plot only.
    # (this is NOT what the lag search uses -- the search uses per-trial scores.
    # this is purely a "does the underlying data look sane" visual check.)
    v1_mean_curve = np.nanmean(v1_scores, axis=1)
    v2_mean_curve = np.nanmean(v2_scores, axis=1)

    return lag_star_ms, best_corr, lags_ms, corrs, time_ms, v1_mean_curve, v2_mean_curve


def plot_pair_diagnostic(pair_key, time_ms, v1_mean_curve, v2_mean_curve,
                          lags_ms, corrs, lag_star_ms, out_dir):
    """
    Two-panel sanity-check plot for ONE session:
      top: raw trial-averaged V1/V2 curves (full time range), with the
           50-250ms analysis window shaded -- lets you eyeball whether the
           underlying data looks reasonable before trusting the lag number.
      bottom: the correlation-vs-lag curve (the green curve from earlier),
              with the winning lag marked -- lets you see whether the peak
              is a clean single bump or a noisy/flat curve where the
              argmax is unstable.
    NOTE: the top panel is trial-AVERAGED purely for visualization -- the
    actual lag search (bottom panel) is computed from per-trial scores,
    never from this averaged curve.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))

    ax = axes[0]
    ax.plot(time_ms, v2_mean_curve, color="#2a78d6", label="V2 (trial-avg score)")
    ax.plot(time_ms, v1_mean_curve, color="#eb6834", linestyle="--", label="V1 (trial-avg score)")
    ax.axvspan(WINDOW_MS[0], WINDOW_MS[1], color="gray", alpha=0.15, label="Analysis window")
    ax.axvline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Time from stimulus onset (ms)")
    ax.set_ylabel("Mean decision score (a.u.)")
    ax.set_title(f"{pair_key}: raw data sanity check")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(lags_ms, corrs, color="#199e70")
    ax2.axvline(lag_star_ms, color="red", linestyle="--", label=f"lag* = {lag_star_ms:+.0f} ms")
    ax2.axvline(0, color="k", linestyle=":", linewidth=1)
    ax2.set_xlabel("Candidate lag (ms)  [positive = V2 leads V1]")
    ax2.set_ylabel("Avg correlation (r)")
    ax2.set_title("Correlation vs. lag")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = Path(out_dir) / f"{pair_key.replace(',', '_')}_diagnostic.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_summary(df, out_dir):
    """
    One bar per session's lag_star_ms, sorted by pair_key (same order as
    the printed table) -- lets you see the "how many agree on direction"
    pattern at a glance, not just as a printed count.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2a78d6" if v > 0 else ("#e34948" if v < 0 else "#898781")
              for v in df["lag_star_ms"]]
    ax.bar(range(len(df)), df["lag_star_ms"], color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["pair_key"], rotation=90, fontsize=7)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_ylabel("lag* (ms)  [positive = V2 leads V1]")
    ax.set_title(f"Per-session lag estimates (n={len(df)})")
    plt.tight_layout()
    out_path = Path(out_dir) / "summary_lags.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_all_pairs(results_root, out_dir=None):
    pairs = find_v1_v2_pairs(results_root)
    rows = []
    curves = {}

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for pair_key, d in sorted(pairs.items()):
        lag_star, best_corr, lags_ms, corrs, time_ms, v1_mean, v2_mean = \
            estimate_pair_lag(d["V1"], d["V2"])
        rows.append({"pair_key": pair_key, "lag_star_ms": lag_star, "best_corr": best_corr})
        curves[pair_key] = (lags_ms, corrs)
        print(f"{pair_key}: lag* = {lag_star:+.0f} ms (r={best_corr:.3f})"
              f"  {'[V2 leads]' if lag_star > 0 else '[V1 leads]' if lag_star < 0 else '[tied]'}")

        if out_dir is not None:
            plot_pair_diagnostic(pair_key, time_ms, v1_mean, v2_mean,
                                  lags_ms, corrs, lag_star, out_dir)

    df = pd.DataFrame(rows)

    if out_dir is not None:
        plot_summary(df, out_dir)
        print(f"\nSaved {len(df)} per-pair diagnostic plots + summary plot to {out_dir.resolve()}")

    return df, curves


def test_group_level(df, tail="greater"):
    """
    One-sample Wilcoxon signed-rank test of lag_star_ms against 0.
    tail='greater' tests median lag > 0 (V2 leads V1 -- the
    hypothesized direction). Uses the exact distribution given n=16.
    """
    lags = df["lag_star_ms"].values
    n = len(lags)
    res = stats.wilcoxon(lags, alternative=tail, mode="exact", zero_method="wilcox")

    print(f"\n=== Group-level test: lag_star_ms vs. 0 (n={n}, tail={tail}) ===")
    print(f"Median lag: {np.median(lags):.1f} ms | Mean lag: {np.mean(lags):.1f} ms")
    print(f"{np.sum(lags > 0)}/{n} pairs show V2 leading V1 (positive lag)")
    print(f"Wilcoxon signed-rank p = {res.pvalue:.4f}")
    return res


if __name__ == "__main__":
    RESULTS_ROOT = r"C:\project\vsdi-face-decoding\results"  # <-- edit if needed
    PLOTS_DIR = Path("v1_v2_lag_plots")  # <-- per-pair + summary plots saved here

    df, curves = run_all_pairs(RESULTS_ROOT, out_dir=PLOTS_DIR)
    print("\n", df.to_string(index=False))

    result = test_group_level(df, tail="greater")

    out_csv = Path("v1_v2_lag_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-pair results to {out_csv.resolve()}")