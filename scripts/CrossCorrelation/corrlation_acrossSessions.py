"""
Method B, all sessions, TWO PHASES — phase 1 (0-130ms) and phase 2 (130-250ms)

Runs the validated single-session Method B pipeline (oof_trial_score-based)
across all 16 V1/V2 (date, session, conditions) pairs, for BOTH phases, and
summarizes:
    - best lag per session, per phase (+ sign: V1-leads vs V2-leads)
    - curve range (max-min r across lags) and flatness flag, per phase
    - overview figures: lag-curves overlaid per phase, histograms of best
      lags per phase, side by side for direct phase-1 vs phase-2 comparison

Sign convention (unchanged):
    positive lag L  ->  V1 sampled at (t + L)  ->  V2 leads V1
    negative lag L  ->  V1 sampled at (t + L), L<0  ->  V1 leads V2

Hypothesis being checked (descriptively, no formal test yet):
    phase 1 (feedforward-dominant): expect V1-leads (negative median lag)
    phase 2 (feedback-dominant):    expect V2-leads (positive median lag)

Still proof-of-concept level -- no Wilcoxon test yet (per earlier decision
to defer formal statistics until the pipeline itself is trusted).
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

PHASE1_T_START, PHASE1_T_STOP = 0, 130
PHASE2_T_START, PHASE2_T_STOP = 130, 250

LAG_MIN = -50
LAG_MAX = 50
LAG_STEP = 10

FLATNESS_RANGE_THRESHOLD = 0.05

V1_DIR = Path(r"C:\project\vsdi-face-decoding\results\V1")
V2_DIR = Path(r"C:\project\vsdi-face-decoding\results\V2")

PHASES = [
    ("phase 1", PHASE1_T_START, PHASE1_T_STOP),
    ("phase 2", PHASE2_T_START, PHASE2_T_STOP),
]

SESSION_SET_1 = {
    "110209_a_1,5", "110209_a_2,4", "110209_c_1,5", "110209_c_2,4",
    "030209_a_1,5", "030209_a_2,4", "030209_f_1,5", "030209_f_2,4",
}
SESSION_SET_2 = {
    "110209_b_1,5", "110209_b_2,4", "110209_d_1,5", "110209_d_2,4",
    "030209_c_1,5", "030209_c_2,4", "030209_e_1,5", "030209_e_2,4",
}
SESSION_SETS = [("set 1", SESSION_SET_1), ("set 2", SESSION_SET_2)]


# ---------------------------------------------------------------------------
# Folder discovery (same as explore_pairs.py / method_b_all_sessions.py)
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
# Method B core (identical to single-session script)
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

    result = {"pair_key": pair_key, "n_trials": n_trials, "time_ms": time_ms}

    # evidence curves (score x true_label) for the grand-average grid figure
    n_half = n_trials // 2
    true_label = np.array([1] * n_half + [-1] * (n_trials - n_half))
    v1_evidence_mean = (v1_scores * true_label[np.newaxis, :]).mean(axis=1)
    v2_evidence_mean = (v2_scores * true_label[np.newaxis, :]).mean(axis=1)
    result["v1_evidence_mean"] = v1_evidence_mean
    result["v2_evidence_mean"] = v2_evidence_mean

    for phase_name, t_start, t_stop in PHASES:
        ref_idx = get_phase_indices(time_ms, t_start, t_stop)
        lags, mean_r = compute_lag_curve(
            v1_scores, v2_scores, n_windows, ref_idx, LAG_MIN, LAG_MAX, LAG_STEP,
        )
        best_lag = int(lags[np.nanargmax(mean_r)])
        best_r = float(np.nanmax(mean_r))
        r_range = float(np.nanmax(mean_r) - np.nanmin(mean_r))
        flat = r_range < FLATNESS_RANGE_THRESHOLD
        result[phase_name] = {
            "lags": lags, "mean_r": mean_r,
            "best_lag": best_lag, "best_r": best_r,
            "r_range": r_range, "flat": flat,
        }
    return result


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
    # Summary table -- both phases side by side
    # -----------------------------------------------------------------
    header = (f"{'pair_key':<18} {'n_trials':>8} "
              f"{'P1_lag':>7} {'P1_r':>7} {'P1_flat':>8}   "
              f"{'P2_lag':>7} {'P2_r':>7} {'P2_flat':>8}")
    print(header)
    for res in all_results:
        p1, p2 = res["phase 1"], res["phase 2"]
        print(f"{res['pair_key']:<18} {res['n_trials']:>8} "
              f"{p1['best_lag']:>+7d} {p1['best_r']:>7.4f} {('FLAT' if p1['flat'] else ''):>8}   "
              f"{p2['best_lag']:>+7d} {p2['best_r']:>7.4f} {('FLAT' if p2['flat'] else ''):>8}")

    # -----------------------------------------------------------------
    # Per-phase, per-SET descriptive summary
    # -----------------------------------------------------------------
    def results_for_set(session_set):
        return [r for r in all_results if r["pair_key"] in session_set]

    unmatched = [r["pair_key"] for r in all_results
                 if r["pair_key"] not in SESSION_SET_1 and r["pair_key"] not in SESSION_SET_2]
    if unmatched:
        print(f"\nWARNING: sessions not in either set: {unmatched}")

    for set_name, session_set in SESSION_SETS:
        set_results = results_for_set(session_set)
        print(f"\n=== {set_name} ({len(set_results)} sessions) ===")
        for phase_name, t_start, t_stop in PHASES:
            best_lags = np.array([r[phase_name]["best_lag"] for r in set_results])
            n_v1_lead = np.sum(best_lags < 0)
            n_v2_lead = np.sum(best_lags > 0)
            n_zero = np.sum(best_lags == 0)
            n_flat = np.sum([r[phase_name]["flat"] for r in set_results])
            print(f"  {phase_name} ({t_start}-{t_stop}ms):")
            print(f"    V1-leads: {n_v1_lead}, V2-leads: {n_v2_lead}, zero: {n_zero}, flat: {n_flat}")
            print(f"    median lag: {np.median(best_lags):+.1f} ms, mean lag: {np.mean(best_lags):+.1f} ms")

    # (also keep the pooled all-16 summary for reference)
    print(f"\n=== all sessions pooled ({len(all_results)}) ===")
    for phase_name, t_start, t_stop in PHASES:
        best_lags = np.array([r[phase_name]["best_lag"] for r in all_results])
        print(f"  {phase_name}: median = {np.median(best_lags):+.1f} ms, mean = {np.mean(best_lags):+.1f} ms")

    # -----------------------------------------------------------------
    # Figure: lag-curves overlaid, SET (rows) x PHASE (cols) grid
    # -----------------------------------------------------------------
    figD, axesD = plt.subplots(len(SESSION_SETS), len(PHASES), figsize=(14, 11), sharey=True, sharex=True)
    for row, (set_name, session_set) in enumerate(SESSION_SETS):
        set_results = results_for_set(session_set)
        for col, (phase_name, t_start, t_stop) in enumerate(PHASES):
            ax = axesD[row, col]
            for res in set_results:
                p = res[phase_name]
                color = "tab:red" if p["flat"] else "tab:gray"
                alpha = 0.4 if p["flat"] else 0.8
                ax.plot(p["lags"], p["mean_r"], marker="o", markersize=3,
                        color=color, alpha=alpha, linewidth=1)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            if row == len(SESSION_SETS) - 1:
                ax.set_xlabel("lag (ms)  [positive = V2 leads V1]")
            ax.set_title(f"{set_name} — {phase_name} ({t_start}-{t_stop}ms)")
        axesD[row, 0].set_ylabel("mean per-trial correlation (r)")
    figD.suptitle("Method B, by session set (red = flagged flat, gray = usable)")
    figD.tight_layout()

    # -----------------------------------------------------------------
    # Figure: histogram of best lags, SET (rows) x PHASE (cols) grid
    # -----------------------------------------------------------------
    figE, axesE = plt.subplots(len(SESSION_SETS), len(PHASES), figsize=(13, 10), sharey=True, sharex=True)
    bins = np.arange(LAG_MIN - LAG_STEP / 2, LAG_MAX + LAG_STEP * 1.5, LAG_STEP)
    for row, (set_name, session_set) in enumerate(SESSION_SETS):
        set_results = results_for_set(session_set)
        for col, (phase_name, t_start, t_stop) in enumerate(PHASES):
            ax = axesE[row, col]
            best_lags = np.array([r[phase_name]["best_lag"] for r in set_results])
            ax.hist(best_lags, bins=bins, color="tab:blue", edgecolor="black")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(np.median(best_lags), color="red", linestyle=":", linewidth=1.5,
                       label=f"median = {np.median(best_lags):+.1f} ms")
            ax.axvline(np.mean(best_lags), color="green", linestyle="-.", linewidth=1.5,
                       label=f"mean = {np.mean(best_lags):+.1f} ms")
            if row == len(SESSION_SETS) - 1:
                ax.set_xlabel("best lag (ms)  [positive = V2 leads V1]")
            ax.set_title(f"{set_name} — {phase_name} ({t_start}-{t_stop}ms)")
            ax.legend(fontsize=8)
        axesE[row, 0].set_ylabel("number of sessions")
    figE.suptitle("Distribution of best lags, by session set")
    figE.tight_layout()

    # -----------------------------------------------------------------
    # Figure: grand-average V1/V2 time courses, one subplot per session
    # -----------------------------------------------------------------
    n_sessions = len(all_results)
    ncols = 4
    nrows = int(np.ceil(n_sessions / ncols))
    figF, axesF = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=True)
    axesF = np.atleast_1d(axesF).flatten()
    for i, res in enumerate(all_results):
        ax = axesF[i]
        ax.plot(res["time_ms"], res["v1_evidence_mean"], color="tab:blue", label="V1", linewidth=1)
        ax.plot(res["time_ms"], res["v2_evidence_mean"], color="tab:orange", label="V2", linewidth=1)
        ax.axvspan(PHASE1_T_START, PHASE1_T_STOP, color="gray", alpha=0.15)
        ax.axvspan(PHASE2_T_START, PHASE2_T_STOP, color="purple", alpha=0.10)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(res["pair_key"], fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n_sessions, len(axesF)):
        axesF[j].axis("off")
    figF.suptitle("Grand-average V1/V2 evidence-toward-correct-class, all sessions\n"
                  "(gray = phase 1, purple = phase 2)")
    figF.tight_layout()

    plt.show()