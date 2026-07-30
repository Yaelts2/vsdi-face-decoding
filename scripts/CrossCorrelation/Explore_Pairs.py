"""
Method B, step 0: rediscover which (date, session, cond1, cond2) pairs have
both a V1 and a V2 sliding-window decoding result on disk, and print basic
stats (trial count, folder path) so we can pick a first test session.

Run this locally where the results/ folders actually live, e.g.:
    C:\\project\\vsdi-face-decoding\\results\\V1
    C:\\project\\vsdi-face-decoding\\results\\V2
"""

import re
import json
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Folder naming pattern (from project recap):
# sliding_window__{date:6digits}{session:letter}{cond1}{cond2}_V{area}_frame{start}-{stop}__SVM_{n_splits}foldCV____{timestamp}
# e.g. sliding_window__030209a15_V2_frame1-100__SVM_10foldCV____2026-07-07_12-16-45
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
    """Parse a results folder name into its component fields, or None if it
    doesn't match the expected pattern."""
    m = FOLDER_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "date": d["date"],
        "session": d["session"],
        "cond1": d["cond1"],
        "cond2": d["cond2"],
        "area": d["area"],
        "start": int(d["start"]),
        "stop": int(d["stop"]),
        "nsplits": int(d["nsplits"]),
        "timestamp": d["timestamp"],
        "folder_name": name,
    }


def scan_area_dir(area_dir):
    """Scan a results/V{area} directory and parse every subfolder found."""
    area_dir = Path(area_dir)
    parsed = []
    if not area_dir.exists():
        print(f"  WARNING: directory does not exist: {area_dir}")
        return parsed
    for sub in sorted(area_dir.iterdir()):
        if not sub.is_dir():
            continue
        info = parse_folder_name(sub.name)
        if info is None:
            print(f"  (skipping unrecognized folder: {sub.name})")
            continue
        info["path"] = sub
        parsed.append(info)
    return parsed


def make_pair_key(info):
    return f"{info['date']}_{info['session']}_{info['cond1']},{info['cond2']}"


def find_v1_v2_pairs(v1_dir, v2_dir):
    """Return a dict: pair_key -> {'V1': info, 'V2': info} for every
    (date, session, cond1, cond2) combo that has BOTH a V1 and V2 result.

    If multiple runs exist for the same key (e.g. re-run with a different
    timestamp), the most recent one (by timestamp string) is kept.
    """
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
    missing_v1 = sorted(set(v2_by_key) - set(v1_by_key))
    missing_v2 = sorted(set(v1_by_key) - set(v2_by_key))

    if missing_v1:
        print(f"Keys with V2 but no V1 (excluded): {missing_v1}")
    if missing_v2:
        print(f"Keys with V1 but no V2 (excluded): {missing_v2}")

    pairs = {}
    for key in common_keys:
        pairs[key] = {"V1": v1_by_key[key], "V2": v2_by_key[key]}
    return pairs


def load_experiment(run_dir):
    """Reload config + results + ROI mask path for one run folder."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    d = np.load(run_dir / "result.npz", allow_pickle=True)
    results = d["results"].item()
    d_roi = np.load(run_dir / "ROI_mask_path.npz", allow_pickle=True)
    ROI_mask_path = str(d_roi["roi_mask_path"])
    return config, results, ROI_mask_path


def summarize_pairs(pairs):
    """Print trial count and other basic stats for each pair, to help pick
    a first test session."""
    print(f"\nFound {len(pairs)} V1/V2 pairs:\n")
    print(f"{'pair_key':<20} {'V1 trials':>10} {'V2 trials':>10} {'V1 n_windows':>13} {'V2 n_windows':>13}")
    for key, entry in pairs.items():
        try:
            _, v1_results, _ = load_experiment(entry["V1"]["path"])
            _, v2_results, _ = load_experiment(entry["V2"]["path"])
            v1_shape = v1_results["oof_trial_score"].shape  # (n_windows, n_trials)
            v2_shape = v2_results["oof_trial_score"].shape
            print(f"{key:<20} {v1_shape[1]:>10} {v2_shape[1]:>10} {v1_shape[0]:>13} {v2_shape[0]:>13}")
        except Exception as e:
            print(f"{key:<20} ERROR loading: {e}")


if __name__ == "__main__":
    V1_DIR = r"C:\project\vsdi-face-decoding\results\V1"
    V2_DIR = r"C:\project\vsdi-face-decoding\results\V2"

    pairs = find_v1_v2_pairs(V1_DIR, V2_DIR)
    summarize_pairs(pairs)


    print("\npair_keys:")
    for key in pairs:
        print(f"  {key}")