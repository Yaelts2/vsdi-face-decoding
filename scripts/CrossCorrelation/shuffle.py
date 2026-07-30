"""
Method B, shuffle control — single session: 110209_a_1,5, phase 1 (0-120ms)

Question this answers: is the lag-curve peak we see real (timing-specific
correspondence between each trial's own V1 and V2), or is it just an
artifact of both areas sharing a generic stimulus-locked temporal profile
(which would make ANY V1/V2 pairing -- even a fake one -- look correlated
near lag=0)?

Method: for each of N_SHUFFLES iterations, circularly shift EACH TRIAL'S
OWN V1 time-course (full 95-window series) by an independent random amount
before recomputing the phase-1 lag curve. This:
    - destroys genuine within-trial V1<->V2 timing correspondence (since
      V1's values are now sourced from the wrong timepoint for that trial)
    - PRESERVES each trial's own temporal autocorrelation / shape (unlike a
      pointwise scramble, which would create an artificially anti-correlated
      null given the smoothness from the sliding window itself)
    - preserves the generic shared-stimulus-locked-shape confound if that's
      what's driving the real curve, since the shifted V1 trace still has
      the same overall stimulus-locked shape, just misaligned in time

If the real curve's peak is genuinely timing-specific, it should sit clearly
above the shuffled-null band, and the "excess" (real - null mean) should be
PEAKED at a plausible lag. If the real curve is just riding the shared
stimulus-locked shape, the null band (built from shuffled, non-corresponding
trial pairings that still share that same shape) should track the real curve
closely at every lag -- flat excess, no real information in the specific lag.
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

LAG_MIN = -50
LAG_MAX = 50
LAG_STEP = 10

N_SHUFFLES = 500
RANDOM_SEED = 0

V1_DIR = Path(r"C:\project\vsdi-face-decoding\results\V1")
V2_DIR = Path(r"C:\project\vsdi-face-decoding\results\V2")

PAIR_KEY = "110209_a_1,5"
V1_FOLDER_HINT = "110209a15_V1"
V2_FOLDER_HINT = "110209a15_V2"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_experiment(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    d = np.load(run_dir / "result.npz", allow_pickle=True)
    results = d["results"].item()
    return config, results


def find_folder(area_dir, hint):
    matches = [p for p in area_dir.iterdir() if p.is_dir() and hint in p.name]
    if len(matches) == 0:
        raise FileNotFoundError(f"No folder in {area_dir} containing '{hint}'")
    if len(matches) > 1:
        matches.sort()
        return matches[-1]
    return matches[0]


# ---------------------------------------------------------------------------
# Method B core
# ---------------------------------------------------------------------------
def centers_to_time_ms(centers):
    return (np.asarray(centers) - ZERO_FRAME) * FRAME_DURATION_MS


def get_phase_indices(time_ms, t_start, t_stop):
    mask = (time_ms >= t_start) & (time_ms <= t_stop)
    return np.where(mask)[0]


def compute_lag_curve(v1_scores, v2_scores, n_windows, ref_idx, lag_min, lag_max, lag_step):
    """Same engine as method_b_poc.py / method_b_all_sessions.py."""
    n_trials = v2_scores.shape[1]
    lags = np.arange(lag_min, lag_max + lag_step, lag_step)
    mean_r_list, surviving_lags = [], []

    for lag in lags:
        shift = int(round(lag / FRAME_DURATION_MS))
        v1_idx = ref_idx + shift
        if v1_idx.min() < 0 or v1_idx.max() > n_windows - 1:
            continue
        v2_vals = v2_scores[ref_idx, :]
        v1_vals = v1_scores[v1_idx, :]
        trial_rs = np.full(n_trials, np.nan)
        for tr in range(n_trials):
            v2_t, v1_t = v2_vals[:, tr], v1_vals[:, tr]
            if np.std(v2_t) == 0 or np.std(v1_t) == 0:
                continue
            r, _ = pearsonr(v2_t, v1_t)
            trial_rs[tr] = r
        mean_r_list.append(np.nanmean(trial_rs))
        surviving_lags.append(lag)

    return np.array(surviving_lags), np.array(mean_r_list)


def circular_shift_per_trial(scores, rng):
    """Circularly shift each trial's (column's) time-course by an
    independent random amount. Preserves each trial's own autocorrelation
    structure; destroys true alignment with the other area."""
    n_windows, n_trials = scores.shape
    shifted = np.empty_like(scores)
    for tr in range(n_trials):
        shift_amt = rng.integers(1, n_windows)  # avoid shift=0 (no-op)
        shifted[:, tr] = np.roll(scores[:, tr], shift_amt)
    return shifted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    v1_folder = find_folder(V1_DIR, V1_FOLDER_HINT)
    v2_folder = find_folder(V2_DIR, V2_FOLDER_HINT)

    v1_config, v1_results = load_experiment(v1_folder)
    v2_config, v2_results = load_experiment(v2_folder)

    v1_scores = v1_results["oof_trial_score"]
    v2_scores = v2_results["oof_trial_score"]
    n_trials = v1_scores.shape[1]
    print(f"{PAIR_KEY}: {n_trials} trials")

    v1_time_ms = centers_to_time_ms(v1_results["centers"])
    v2_time_ms = centers_to_time_ms(v2_results["centers"])
    assert np.array_equal(v1_time_ms, v2_time_ms), "V1/V2 centers differ -- stopping"
    time_ms = v1_time_ms
    n_windows = len(time_ms)

    ref_idx = get_phase_indices(time_ms, PHASE1_T_START, PHASE1_T_STOP)

    # ---------------------------------------------------------------
    # Real curve
    # ---------------------------------------------------------------
    real_lags, real_r = compute_lag_curve(
        v1_scores, v2_scores, n_windows, ref_idx, LAG_MIN, LAG_MAX, LAG_STEP,
    )
    real_best_idx = np.nanargmax(real_r)
    real_best_lag = int(real_lags[real_best_idx])
    real_best_r = float(real_r[real_best_idx])
    print(f"Real curve: best lag = {real_best_lag:+d} ms, r = {real_best_r:.4f}")

    # ---------------------------------------------------------------
    # Null distribution: shuffle V1 (circular shift per trial), rebuild
    # curve, repeat N_SHUFFLES times
    # ---------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    null_curves = np.full((N_SHUFFLES, len(real_lags)), np.nan)

    for i in range(N_SHUFFLES):
        v1_shuffled = circular_shift_per_trial(v1_scores, rng)
        shuf_lags, shuf_r = compute_lag_curve(
            v1_shuffled, v2_scores, n_windows, ref_idx, LAG_MIN, LAG_MAX, LAG_STEP,
        )
        # shuf_lags should match real_lags exactly (same bounds check every time)
        assert np.array_equal(shuf_lags, real_lags), "lag grid mismatch between real and shuffled run"
        null_curves[i, :] = shuf_r
        if (i + 1) % 100 == 0:
            print(f"  shuffle {i+1}/{N_SHUFFLES} done")

    null_mean = np.nanmean(null_curves, axis=0)
    null_p2_5 = np.nanpercentile(null_curves, 2.5, axis=0)
    null_p97_5 = np.nanpercentile(null_curves, 97.5, axis=0)

    excess = real_r - null_mean

    # p-value at the real best lag: fraction of shuffles whose curve at that
    # SAME lag meets/exceeds the real r there
    p_at_best_lag = np.mean(null_curves[:, real_best_idx] >= real_best_r)

    # p-value for "best-of-shuffle" comparison: fraction of shuffles whose
    # OWN best-across-all-lags r meets/exceeds the real best r (more
    # conservative / standard for a max-over-lags statistic)
    shuffle_maxes = np.nanmax(null_curves, axis=1)
    p_max_comparison = np.mean(shuffle_maxes >= real_best_r)

    print(f"\nNull mean at real best lag ({real_best_lag:+d}ms): {null_mean[real_best_idx]:.4f}")
    print(f"Excess (real - null_mean) at best lag: {excess[real_best_idx]:.4f}")
    print(f"p-value (pointwise, same lag): {p_at_best_lag:.4f}")
    print(f"p-value (max-over-lags, conservative): {p_max_comparison:.4f}")
    print(f"Excess curve range (max-min across lags): {np.nanmax(excess) - np.nanmin(excess):.4f}")

    # ---------------------------------------------------------------
    # Figure F: real curve vs null band
    # ---------------------------------------------------------------
    figF, axF = plt.subplots(figsize=(8, 5))
    axF.fill_between(real_lags, null_p2_5, null_p97_5, color="gray", alpha=0.3,
                      label="null 95% band (shuffled)")
    axF.plot(real_lags, null_mean, color="gray", linestyle="--", label="null mean")
    axF.plot(real_lags, real_r, color="tab:blue", marker="o", label="real curve")
    axF.axvline(0, color="black", linewidth=0.5)
    axF.axvline(real_best_lag, color="red", linestyle=":", linewidth=1.5,
                label=f"real best lag = {real_best_lag:+d}ms")
    axF.set_xlabel("lag (ms)  [positive = V2 leads V1]")
    axF.set_ylabel("mean per-trial correlation (r)")
    axF.set_title(f"Real vs. shuffled-null curve — {PAIR_KEY}\n"
                   f"p(pointwise)={p_at_best_lag:.3f}, p(max-comparison)={p_max_comparison:.3f}")
    axF.legend(fontsize=8)
    figF.tight_layout()

    # ---------------------------------------------------------------
    # Figure G: excess curve (real - null mean) -- flat vs peaked diagnosis
    # ---------------------------------------------------------------
    figG, axG = plt.subplots(figsize=(8, 4))
    axG.plot(real_lags, excess, color="tab:purple", marker="o")
    axG.axhline(0, color="black", linewidth=0.5)
    axG.axvline(0, color="gray", linestyle="--", linewidth=1)
    axG.set_xlabel("lag (ms)  [positive = V2 leads V1]")
    axG.set_ylabel("excess r (real - null mean)")
    axG.set_title(f"Excess curve — {PAIR_KEY}\n"
                   f"(flat = generic shared shape; peaked = timing-specific signal)")
    figG.tight_layout()

    plt.show()