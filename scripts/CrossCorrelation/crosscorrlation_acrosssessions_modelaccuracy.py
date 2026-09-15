"""
Method B, all sessions — phase 1 (0-120ms post-stimulus)

Runs the validated single-session Method B pipeline (see method_b_poc.py)
across all 16 V1/V2 (date, session, conditions) pairs, and summarizes:
    - best lag per session (+ sign: V1-leads vs V2-leads)
    - curve range (max-min r across lags) and flatness flag per session
    - overview figures: all 16 lag-curves overlaid, histogram of best lags

Sign convention (unchanged):
    positive lag L  ->  V1 sampled at (t + L)  ->  V2 leads V1
    negative lag L  ->  V1 sampled at (t + L), L<0  ->  V1 leads V2

NOTE: this is still proof-of-concept level -- no Wilcoxon test yet (per
earlier decision to defer formal statistics). This script just lays all 16
curves + best lags out so we can eyeball whether there's a consistent
pattern before formalizing anything.
"""

import re
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

PHASE1_T_START = 130
PHASE1_T_STOP = 250

LAG_MIN = -50
LAG_MAX = 50
LAG_STEP = 10

FLATNESS_RANGE_THRESHOLD = 0.05

V1_DIR = Path(r"C:\project\vsdi-face-decoding\results\V1")
V2_DIR = Path(r"C:\project\vsdi-face-decoding\results\V2")


# ---------------------------------------------------------------------------
# Folder discovery (rebuilt from recap, same pattern as explore_pairs.py)
# ---------------------------------------------------------------------------
FOLDER_RE = re.compile(
    r"^sliding_window__"
    r"(?P<date>\d{6})"
    r"(?P<session>[a-zA-Z])"
    r"(?P<cond1>\d)"
    r"(?P<cond2>\d)"
    r"_V(?P<area>\d)"
    r"_frame(?P<start>\d+)-(?P<stop>\d+)"
    r"__SVM_(?P<nsplits>\d+)foldCV____"
    r"(?P<timestamp>.+)$"
)


def parse_folder_name(name):
    m = FOLDER_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "date": d["date"], "session": d["session"],
        "cond1": d["cond1"], "cond2": d["cond2"], "area": d["area"],
        "timestamp": d["timestamp"], "folder_name": name,
    }


def scan_area_dir(area_dir):
    area_dir = Path(area_dir)
    parsed = []
    for sub in sorted(area_dir.iterdir()):
        if not sub.is_dir():
            continue
        info = parse_folder_name(sub.name)
        if info is not None:
            info["path"] = sub
            parsed.append(info)
    return parsed


def make_pair_key(info):
    return f"{info['date']}_{info['session']}_{info['cond1']},{info['cond2']}"


def find_v1_v2_pairs(v1_dir, v2_dir):
    v1_entries = scan_area_dir(v1_dir)
    v2_entries = scan_area_dir(v2_dir)

    def latest_per_key(entries):
        by_key = {}
        for info in entries:
            key = make_pair_key(info)
            if key not in by_key or info["timestamp"] > by_key[key]["timestamp"]:
                by_key[key] = info
        return by_key

    v1_by_key = latest_per_key(v1_entries)
    v2_by_key = latest_per_key(v2_entries)
    common_keys = sorted(set(v1_by_key) & set(v2_by_key))
    return {k: {"V1": v1_by_key[k], "V2": v2_by_key[k]} for k in common_keys}


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


# ---------------------------------------------------------------------------
# Method B core (identical to method_b_poc.py)
# ---------------------------------------------------------------------------
def centers_to_time_ms(centers):
    return (np.asarray(centers) - ZERO_FRAME) * FRAME_DURATION_MS


def get_phase_indices(time_ms, t_start, t_stop):
    mask = (time_ms >= t_start) & (time_ms <= t_stop)
    return np.where(mask)[0]


def compute_lag_curve(v1_scores, v2_scores, n_windows, ref_idx, lag_min, lag_max, lag_step):
    index_step_ms = FRAME_DURATION_MS
    n_trials = v2_scores.shape[1]
    lags = np.arange(lag_min, lag_max + lag_step, lag_step)
    mean_r_list, surviving_lags = [], []

    for lag in lags:
        shift = int(round(lag / index_step_ms))
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


def run_one_session(pair_key, entry):
    v1_config, v1_results = load_experiment(entry["V1"]["path"])
    v2_config, v2_results = load_experiment(entry["V2"]["path"])

    v1_scores = v1_results["oof_trial_score"]
    v2_scores = v2_results["oof_trial_score"]
    n_trials = v1_scores.shape[1]

    v1_time_ms = centers_to_time_ms(v1_results["centers"])
    v2_time_ms = centers_to_time_ms(v2_results["centers"])
    if not np.array_equal(v1_time_ms, v2_time_ms):
        raise ValueError(f"{pair_key}: V1/V2 centers differ -- shared-centers assumption broken")
    time_ms = v1_time_ms
    n_windows = len(time_ms)

    ref_idx = get_phase_indices(time_ms, PHASE1_T_START, PHASE1_T_STOP)

    lags, mean_r = compute_lag_curve(
        v1_scores, v2_scores, n_windows, ref_idx, LAG_MIN, LAG_MAX, LAG_STEP,
    )

    best_lag = int(lags[np.nanargmax(mean_r)])
    best_r = float(np.nanmax(mean_r))
    r_range = float(np.nanmax(mean_r) - np.nanmin(mean_r))
    flat = r_range < FLATNESS_RANGE_THRESHOLD

    return {
        "pair_key": pair_key,
        "n_trials": n_trials,
        "lags": lags,
        "mean_r": mean_r,
        "best_lag": best_lag,
        "best_r": best_r,
        "r_range": r_range,
        "flat": flat,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pairs = find_v1_v2_pairs(V1_DIR, V2_DIR)
    print(f"Found {len(pairs)} pairs.\n")

    all_results = []
    for pair_key, entry in pairs.items():
        try:
            res = run_one_session(pair_key, entry)
            all_results.append(res)
        except Exception as e:
            print(f"  {pair_key}: ERROR -- {e}")

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    print(f"\n{'pair_key':<18} {'n_trials':>8} {'best_lag':>9} {'best_r':>8} {'r_range':>8} {'flat?':>6}")
    for res in all_results:
        flag = "FLAT" if res["flat"] else ""
        print(f"{res['pair_key']:<18} {res['n_trials']:>8} {res['best_lag']:>+9d} "
              f"{res['best_r']:>8.4f} {res['r_range']:>8.4f} {flag:>6}")

    best_lags = np.array([r["best_lag"] for r in all_results])
    n_v1_lead = np.sum(best_lags < 0)
    n_v2_lead = np.sum(best_lags > 0)
    n_zero = np.sum(best_lags == 0)
    n_flat = np.sum([r["flat"] for r in all_results])

    print(f"\n{len(all_results)} sessions total")
    print(f"  V1-leads (lag<0): {n_v1_lead}")
    print(f"  V2-leads (lag>0): {n_v2_lead}")
    print(f"  zero lag: {n_zero}")
    print(f"  flagged as flat curve: {n_flat}")
    print(f"  median best lag: {np.median(best_lags):+.1f} ms")

    # -----------------------------------------------------------------
    # Figure D: all 16 lag-curves overlaid
    # -----------------------------------------------------------------
    figD, axD = plt.subplots(figsize=(8, 6))
    for res in all_results:
        color = "tab:red" if res["flat"] else "tab:gray"
        alpha = 0.4 if res["flat"] else 0.8
        axD.plot(res["lags"], res["mean_r"], marker="o", markersize=3,
                  color=color, alpha=alpha, linewidth=1,
                  label=res["pair_key"] if not res["flat"] else None)
    axD.axvline(0, color="black", linestyle="--", linewidth=1)
    axD.set_xlabel("lag (ms)  [positive = V2 leads V1]")
    axD.set_ylabel("mean per-trial correlation (r)")
    axD.set_title(f"Method B, phase 1 (0-{PHASE1_T_STOP}ms) — all sessions\n"
                   f"(red = flagged flat, gray = usable)")
    figD.tight_layout()

    # -----------------------------------------------------------------
    # Figure E: histogram of best lags across sessions
    # -----------------------------------------------------------------
    figE, axE = plt.subplots(figsize=(7, 4))
    bins = np.arange(LAG_MIN - LAG_STEP / 2, LAG_MAX + LAG_STEP * 1.5, LAG_STEP)
    axE.hist(best_lags, bins=bins, color="tab:blue", edgecolor="black")
    axE.axvline(0, color="black", linestyle="--", linewidth=1)
    axE.axvline(np.median(best_lags), color="red", linestyle=":", linewidth=1.5,
                label=f"median = {np.median(best_lags):+.1f} ms")
    axE.set_xlabel("best lag (ms)  [positive = V2 leads V1]")
    axE.set_ylabel("number of sessions")
    axE.set_title(f"Distribution of best lags across {len(all_results)} sessions")
    axE.legend()
    figE.tight_layout()

    plt.show()