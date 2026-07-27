"""
shuffle_control.py

Trial-shuffle control for the V1-V2 lag cross-correlation method.

For each session: recompute the correlation-vs-lag curve many times
after randomly permuting which V1 trial is paired with which V2 trial
(breaking the TRUE trial-by-trial correspondence while leaving each
area's own signal completely intact). This builds a null band showing
what the curve looks like when trial pairing carries no real
information.

TWO SEPARATE THINGS TO CHECK IN THE OUTPUT -- both matter, neither
alone is sufficient (see validation notes below):

1. p_peak: is the REAL curve's peak higher than the shuffled null's
   peak more often than chance? If not (large p), the observed
   correlation doesn't require true trial correspondence at all --
   serious problem.

2. Shape of the "excess" curve (real minus null mean), across lags:
   - FLAT and elevated at every lag (including implausible ones like
     +-150ms) -> generic, non-timing-specific trial-level coupling
     (e.g. shared arousal/motion). This CAN be significant (low
     p_peak) while carrying zero information about V1-V2 timing.
   - PEAKED at a specific, plausible lag, higher there than elsewhere
     -> genuine timing-locked coupling. This is what would actually
     support the "V2 leads V1" hypothesis.

   Validated on synthetic data: a purely generic shared-trial-gain
   scenario (no timing relationship at all, built by construction)
   passed the p_peak significance test (p=0.003) just as easily as a
   genuine 40ms-lag scenario did -- the p-value alone CANNOT tell these
   apart. Only the flat-vs-peaked shape does.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_v2_lag_crosscorr import (
    ZERO_FRAME, FRAME_DURATION_MS, WINDOW_MS, LAG_RANGE_MS, LAG_STEP_MS,
    load_experiment, find_v1_v2_pairs, compute_avg_corr_curve,
)

RESULTS_ROOT = r"C:\project\vsdi-face-decoding\results"  # <-- edit if needed
PAIR_KEYS = None   # <-- e.g. ["110209_a_1,5", "030209_f_1,5"] ; None = ALL 16 pairs
N_SHUFFLES = 500
SEED = 0
OUT_DIR = Path("shuffle_control_plots")


def run_shuffle_control(v1_scores, v2_scores, time_ms, lags_ms, rng, n_shuffles):
    window_mask = (time_ms >= WINDOW_MS[0]) & (time_ms <= WINDOW_MS[1])
    window_idx = np.where(window_mask)[0]
    n_trials = v1_scores.shape[1]

    real_curve = compute_avg_corr_curve(v1_scores, v2_scores, window_idx, lags_ms)
    real_peak = np.nanmax(real_curve)
    real_peak_lag = lags_ms[np.nanargmax(real_curve)]

    null_curves = np.full((n_shuffles, len(lags_ms)), np.nan)
    for s in range(n_shuffles):
        perm = rng.permutation(n_trials)
        null_curves[s] = compute_avg_corr_curve(v1_scores[:, perm], v2_scores, window_idx, lags_ms)

    null_peaks = np.nanmax(null_curves, axis=1)
    p_peak = (np.sum(null_peaks >= real_peak) + 1) / (n_shuffles + 1)

    null_mean = np.nanmean(null_curves, axis=0)
    excess_curve = real_curve - null_mean
    # crude flatness index: spread of the excess curve relative to its own peak.
    # near 1 = sharply peaked at one lag; near 0 = flat/uniform elevation everywhere.
    excess_range = np.nanmax(excess_curve) - np.nanmin(excess_curve)
    flatness_ratio = 1 - (excess_range / (np.nanmax(excess_curve) + 1e-9)) if np.nanmax(excess_curve) > 0 else np.nan

    return {
        "lags_ms": lags_ms,
        "real_curve": real_curve,
        "real_peak": real_peak,
        "real_peak_lag": real_peak_lag,
        "null_curves": null_curves,
        "null_mean": null_mean,
        "excess_curve": excess_curve,
        "p_peak": p_peak,
        "flatness_ratio": flatness_ratio,
    }


def plot_shuffle_control(pair_key, res, out_dir):
    lags_ms = res["lags_ms"]
    null_curves = res["null_curves"]
    lo = np.nanpercentile(null_curves, 2.5, axis=0)
    hi = np.nanpercentile(null_curves, 97.5, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax = axes[0]
    ax.fill_between(lags_ms, lo, hi, color="gray", alpha=0.3, label="Shuffled null (95% band)")
    ax.plot(lags_ms, res["null_mean"], color="gray", linestyle=":", linewidth=1, label="Shuffled null (mean)")
    ax.plot(lags_ms, res["real_curve"], color="#199e70", linewidth=2, label="Real (unshuffled)")
    ax.axvline(res["real_peak_lag"], color="red", linestyle="--", linewidth=1,
               label=f"real peak lag = {res['real_peak_lag']:+.0f}ms")
    ax.axvline(0, color="k", linestyle=":", linewidth=1)
    ax.set_ylabel("Avg correlation (r)")
    ax.set_title(f"{pair_key}: real vs. shuffled-trial null   p(peak)={res['p_peak']:.3f}")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(lags_ms, res["excess_curve"], color="#993c1d", linewidth=2)
    ax2.axhline(0, color="k", linestyle=":", linewidth=1)
    ax2.axvline(res["real_peak_lag"], color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Candidate lag (ms)  [positive = V2 leads V1]")
    ax2.set_ylabel("Excess corr.\n(real - null mean)")
    flat_txt = "FLAT (generic coupling, not timing-specific)" if res["flatness_ratio"] > 0.7 \
        else "PEAKED (timing-specific signal)" if res["flatness_ratio"] < 0.3 \
        else "ambiguous"
    ax2.set_title(f"Excess correlation shape: {flat_txt}  (flatness_ratio={res['flatness_ratio']:.2f})")

    plt.tight_layout()
    out_path = Path(out_dir) / f"{pair_key.replace(',', '_')}_shuffle_control.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    pairs = find_v1_v2_pairs(RESULTS_ROOT)
    keys = PAIR_KEYS or sorted(pairs.keys())
    lags_ms = np.arange(-LAG_RANGE_MS, LAG_RANGE_MS + 1, LAG_STEP_MS)
    rng = np.random.default_rng(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for pair_key in keys:
        d = pairs[pair_key]
        _, v1_results, _ = load_experiment(d["V1"])
        _, v2_results, _ = load_experiment(d["V2"])
        v1_scores = v1_results["oof_trial_score"]
        v2_scores = v2_results["oof_trial_score"]
        centers = np.asarray(v1_results["centers"])
        time_ms = (centers - ZERO_FRAME) * FRAME_DURATION_MS

        res = run_shuffle_control(v1_scores, v2_scores, time_ms, lags_ms, rng, N_SHUFFLES)
        plot_shuffle_control(pair_key, res, OUT_DIR)

        flat_tag = "FLAT" if res["flatness_ratio"] > 0.7 else "PEAKED" if res["flatness_ratio"] < 0.3 else "ambiguous"
        print(f"{pair_key}: real peak r={res['real_peak']:.3f} at lag={res['real_peak_lag']:+.0f}ms, "
              f"p(peak)={res['p_peak']:.4f}, shape={flat_tag} (flatness={res['flatness_ratio']:.2f})")

        summary_rows.append({
            "pair_key": pair_key,
            "real_peak": res["real_peak"],
            "real_peak_lag": res["real_peak_lag"],
            "p_peak": res["p_peak"],
            "flatness_ratio": res["flatness_ratio"],
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_DIR / "shuffle_control_summary.csv", index=False)
    print(f"\nSaved plots + summary to {OUT_DIR.resolve()}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()