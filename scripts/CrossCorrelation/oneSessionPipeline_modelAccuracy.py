"""
Method B, proof of concept — single session: 110209_a_1,5

Question: within each phase, does each trial's own V1 time-course look like
a time-shifted copy of that SAME trial's own V2 time-course?
    phase 1: 0-130ms post-stimulus (feedforward-dominant)
    phase 2: 130-250ms post-stimulus (feedback-dominant)

Sign convention (same as Method A):
    positive lag L  ->  V1 sampled at (t + L)  ->  V2 leads V1
    negative lag L  ->  V1 sampled at (t + L), L<0, i.e. earlier V1  ->  V1 leads V2

KEY SIMPLIFICATION vs. earlier draft: all sessions use identical sliding-
window params (window_size=5, step=1, 95 centers), so V1 and V2 share the
exact same `centers` / `time_ms` array for a given pair. A lag in ms is
therefore just an integer shift in array-index units (10ms per step) --
no nearest-neighbor snapping or tolerance matching needed anywhere.

Pipeline (run once per phase, using the SAME loaded oof_trial_score):
    1. Load V1 + V2 oof_trial_score and centers for this pair.
    2. Convert centers -> time_ms (zero_frame=27, frame_duration_ms=10) and
       confirm V1/V2 time_ms are identical (sanity check).
    3. Find the V2 window-indices for this phase.
    4. For each lag (integer number of 10ms steps), shift those indices to
       get the matching V1 window-indices. Drop a lag only if the shift
       pushes an index outside [0, n_windows-1].
    5. For each trial: Pearson r between the V2 values and the lag-shifted
       V1 values across timepoints.
    6. Average r across trials -> one correlation-vs-lag curve.
    7. Flag if the curve is nearly flat across lags (best lag not
       meaningfully distinguishable from its neighbors).
    8. Figures: (A) grand-average V1/V2 time courses with BOTH phase
       windows highlighted, (B) per-lag overlay grid per phase (two
       figures), (C) main lag-vs-correlation curves for both phases as
       side-by-side subplots in one figure.
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
PHASE1_T_STOP = 130

PHASE2_T_START = 130
PHASE2_T_STOP = 250

LAG_MIN = -50
LAG_MAX = 50
LAG_STEP = 10  # must be a multiple of FRAME_DURATION_MS

FLATNESS_RANGE_THRESHOLD = 0.05  # if max(r) - min(r) across lags < this, flag it

V1_DIR = Path(r"C:\project\vsdi-face-decoding\results\V1")
V2_DIR = Path(r"C:\project\vsdi-face-decoding\results\V2")

PAIR_KEY = "030209_f_2,4"
V1_FOLDER_HINT = "030209f24_V1"
V2_FOLDER_HINT = "030209f24_V2"


# ---------------------------------------------------------------------------
# Loading (rebuilt from recap)
# ---------------------------------------------------------------------------
def load_experiment(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    d = np.load(run_dir / "result.npz", allow_pickle=True)
    results = d["results"].item()
    d_roi = np.load(run_dir / "ROI_mask_path.npz", allow_pickle=True)
    ROI_mask_path = str(d_roi["roi_mask_path"])
    return config, results, ROI_mask_path


def find_folder(area_dir, hint):
    matches = [p for p in area_dir.iterdir() if p.is_dir() and hint in p.name]
    if len(matches) == 0:
        raise FileNotFoundError(f"No folder in {area_dir} containing '{hint}'")
    if len(matches) > 1:
        matches.sort()
        print(f"  multiple matches for '{hint}', using most recent: {matches[-1].name}")
        return matches[-1]
    return matches[0]


# ---------------------------------------------------------------------------
# Method B core
# ---------------------------------------------------------------------------
def centers_to_time_ms(centers):
    return (np.asarray(centers) - ZERO_FRAME) * FRAME_DURATION_MS


def get_phase_indices(time_ms, t_start, t_stop):
    """Indices of all windows whose time_ms falls in [t_start, t_stop]."""
    mask = (time_ms >= t_start) & (time_ms <= t_stop)
    return np.where(mask)[0]


def compute_lag_curve(v1_scores, v2_scores, n_windows, ref_idx, lag_min, lag_max, lag_step):
    """
    v1_scores, v2_scores: shape (n_windows, n_trials), positionally aligned
    by trial and by window-index (identical centers for both areas).
    ref_idx: array of V2 window-indices used as the fixed reference points.

    Returns: lags (array, ms), mean_r (array), per_trial_r (dict lag->array),
             v1_idx_by_lag (dict lag -> array of V1 indices used)
    """
    index_step_ms = FRAME_DURATION_MS  # ms per single window-index step
    n_trials = v2_scores.shape[1]
    assert v1_scores.shape[1] == n_trials, "V1/V2 trial count mismatch!"

    lags = np.arange(lag_min, lag_max + lag_step, lag_step)
    mean_r_list = []
    surviving_lags = []
    per_trial_r = {}
    v1_idx_by_lag = {}

    for lag in lags:
        shift = int(round(lag / index_step_ms))  # index shift for this lag
        v1_idx = ref_idx + shift
        if v1_idx.min() < 0 or v1_idx.max() > n_windows - 1:
            print(f"  lag={lag:+d}ms: dropped (shifted index out of range [0,{n_windows-1}])")
            continue

        v2_vals = v2_scores[ref_idx, :]  # (n_ref, n_trials)
        v1_vals = v1_scores[v1_idx, :]   # (n_ref, n_trials)

        trial_rs = np.full(n_trials, np.nan)
        for tr in range(n_trials):
            v2_t = v2_vals[:, tr]
            v1_t = v1_vals[:, tr]
            if np.std(v2_t) == 0 or np.std(v1_t) == 0:
                continue
            r, _ = pearsonr(v2_t, v1_t)
            trial_rs[tr] = r

        mean_r = np.nanmean(trial_rs)
        mean_r_list.append(mean_r)
        surviving_lags.append(lag)
        per_trial_r[lag] = trial_rs
        v1_idx_by_lag[lag] = v1_idx

    return np.array(surviving_lags), np.array(mean_r_list), per_trial_r, v1_idx_by_lag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    v1_folder = find_folder(V1_DIR, V1_FOLDER_HINT)
    v2_folder = find_folder(V2_DIR, V2_FOLDER_HINT)
    print(f"V1 folder: {v1_folder.name}")
    print(f"V2 folder: {v2_folder.name}")

    v1_config, v1_results, _ = load_experiment(v1_folder)
    v2_config, v2_results, _ = load_experiment(v2_folder)

    v1_scores = v1_results["oof_trial_score"]  # (n_windows, n_trials)
    v2_scores = v2_results["oof_trial_score"]
    print(f"V1 scores shape: {v1_scores.shape}, V2 scores shape: {v2_scores.shape}")

    # ---------------------------------------------------------------
    # True labels: first half of trials = face (+1), second half = nonface (-1).
    # Used ONLY for plotting "evidence toward correct class" (Fig A/B), so
    # opposite-signed-but-both-correct trials don't cancel in the average.
    # The Method B correlation itself uses raw scores, unaffected by sign.
    # ---------------------------------------------------------------
    n_trials = v1_scores.shape[1]
    assert n_trials % 2 == 0, f"expected an even trial count (half face/half nonface), got {n_trials}"
    n_half = n_trials // 2
    true_label = np.array([1] * n_half + [-1] * n_half)
    print(f"Trial labels: {n_half} face (+1), {n_half} nonface (-1)")

    v1_time_ms = centers_to_time_ms(v1_results["centers"])
    v2_time_ms = centers_to_time_ms(v2_results["centers"])

    # sanity check: centers should be identical across areas
    if not np.array_equal(v1_time_ms, v2_time_ms):
        raise ValueError(
            "V1 and V2 centers differ for this pair -- the 'shared centers' "
            "assumption doesn't hold here. Stopping so this can be checked "
            "manually rather than silently falling back to approximate matching."
        )
    time_ms = v1_time_ms
    n_windows = len(time_ms)
    print(f"Shared time range: {time_ms.min():.0f} to {time_ms.max():.0f} ms "
          f"({n_windows} windows, {FRAME_DURATION_MS}ms/step)")

    v1_evidence = v1_scores * true_label[np.newaxis, :]  # (n_windows, n_trials)
    v2_evidence = v2_scores * true_label[np.newaxis, :]

    # ---------------------------------------------------------------
    # Run Method B for phase 1 and phase 2, on the SAME loaded scores
    # ---------------------------------------------------------------
    phases = [
        ("phase 1", PHASE1_T_START, PHASE1_T_STOP),
        ("phase 2", PHASE2_T_START, PHASE2_T_STOP),
    ]
    phase_results = {}

    for phase_name, t_start, t_stop in phases:
        ref_idx = get_phase_indices(time_ms, t_start, t_stop)
        ref_times = time_ms[ref_idx]
        print(f"\n{phase_name} ({t_start}-{t_stop}ms) reference timepoints (ms): "
              f"{ref_times}  ({len(ref_idx)} points)")

        lags, mean_r, per_trial_r, v1_idx_by_lag = compute_lag_curve(
            v1_scores, v2_scores, n_windows, ref_idx, LAG_MIN, LAG_MAX, LAG_STEP,
        )

        print(f"{'lag (ms)':>10} {'mean r':>10}")
        for lag, r in zip(lags, mean_r):
            marker = "  <-- max" if r == np.nanmax(mean_r) else ""
            print(f"{lag:>10} {r:>10.4f}{marker}")

        best_lag = lags[np.nanargmax(mean_r)]
        r_range = np.nanmax(mean_r) - np.nanmin(mean_r)
        print(f"{phase_name}: best lag = {best_lag:+d} ms "
              f"({'V2 leads V1' if best_lag > 0 else 'V1 leads V2' if best_lag < 0 else 'no lead'})")
        print(f"{phase_name}: curve range (max-min r across lags): {r_range:.4f}")
        if r_range < FLATNESS_RANGE_THRESHOLD:
            print(f"*** FLAG ({phase_name}): curve is nearly flat (range < {FLATNESS_RANGE_THRESHOLD}) "
                  f"-- best lag is not clearly distinguishable from neighboring lags. ***")

        phase_results[phase_name] = {
            "t_start": t_start, "t_stop": t_stop,
            "ref_idx": ref_idx, "ref_times": ref_times,
            "lags": lags, "mean_r": mean_r,
            "per_trial_r": per_trial_r, "v1_idx_by_lag": v1_idx_by_lag,
            "best_lag": best_lag, "r_range": r_range,
        }

    # -----------------------------------------------------------------
    # Figure A: grand-average V1/V2 time courses, BOTH phase windows shaded
    # -----------------------------------------------------------------
    figA, axA = plt.subplots(figsize=(9, 4))
    axA.plot(time_ms, v1_evidence.mean(axis=1), label="V1 (grand avg, evidence toward correct class)", color="tab:blue")
    axA.plot(time_ms, v2_evidence.mean(axis=1), label="V2 (grand avg, evidence toward correct class)", color="tab:orange")
    axA.axvspan(PHASE1_T_START, PHASE1_T_STOP, color="gray", alpha=0.15, label="phase 1 window")
    axA.axvspan(PHASE2_T_START, PHASE2_T_STOP, color="purple", alpha=0.10, label="phase 2 window")
    axA.axhline(0, color="black", linewidth=0.5)
    axA.set_xlabel("time (ms)")
    axA.set_ylabel("mean (score x true_label)\n[>0 = correct-direction confidence]")
    axA.set_title(f"Grand-average V1/V2 evidence-toward-correct-class — {PAIR_KEY}")
    axA.legend(fontsize=8)
    figA.tight_layout()

    # -----------------------------------------------------------------
    # Figure B: per-lag overlay grid -- one figure PER PHASE
    # -----------------------------------------------------------------
    for phase_name, t_start, t_stop in phases:
        res = phase_results[phase_name]
        lags, mean_r = res["lags"], res["mean_r"]
        ref_idx, ref_times = res["ref_idx"], res["ref_times"]
        v1_idx_by_lag, best_lag = res["v1_idx_by_lag"], res["best_lag"]

        n_lags_shown = len(lags)
        ncols = 4
        nrows = int(np.ceil(n_lags_shown / ncols))
        figB, axesB = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
        axesB = np.atleast_1d(axesB).flatten()
        v2_ref_mean = v2_evidence[ref_idx, :].mean(axis=1)
        for i, lag in enumerate(lags):
            ax = axesB[i]
            v1_idx = v1_idx_by_lag[lag]
            v1_shifted_mean = v1_evidence[v1_idx, :].mean(axis=1)
            ax.plot(ref_times, v2_ref_mean, "o-", color="tab:orange", label="V2(t)")
            ax.plot(ref_times, v1_shifted_mean, "o-", color="tab:blue", label=f"V1(t{lag:+d})")
            title_color = "red" if lag == best_lag else "black"
            title_suffix = "  <-- BEST" if lag == best_lag else ""
            ax.set_title(f"lag={lag:+d}ms, r={mean_r[i]:.3f}{title_suffix}", fontsize=10, color=title_color)
            ax.axhline(0, color="black", linewidth=0.5)
            if lag == best_lag:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(n_lags_shown, len(axesB)):
            axesB[j].axis("off")
        figB.suptitle(f"Method B alignment per lag, {phase_name} ({t_start}-{t_stop}ms) — {PAIR_KEY}")
        figB.tight_layout()

    # -----------------------------------------------------------------
    # Figure C: main result -- mean r vs lag, BOTH phases as subplots
    # -----------------------------------------------------------------
    figC, axesC = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (phase_name, t_start, t_stop) in zip(axesC, phases):
        res = phase_results[phase_name]
        ax.plot(res["lags"], res["mean_r"], marker="o")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(res["best_lag"], color="red", linestyle=":", linewidth=1.5,
                   label=f"best lag = {res['best_lag']:+d} ms")
        ax.set_xlabel("lag (ms)  [positive = V2 leads V1]")
        title = f"{phase_name} ({t_start}-{t_stop}ms)"
        if res["r_range"] < FLATNESS_RANGE_THRESHOLD:
            title += "\n(FLAGGED: curve nearly flat)"
        ax.set_title(title)
        ax.legend()
    axesC[0].set_ylabel("mean per-trial correlation (r)")
    figC.suptitle(f"Method B, phase 1 vs phase 2 — {PAIR_KEY}")
    figC.tight_layout()

    plt.show()