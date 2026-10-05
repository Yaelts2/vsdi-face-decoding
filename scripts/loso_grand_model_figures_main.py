"""
loso_grand_model_figures_main.py

Open the saved nested leave-one-face-out model (.npz written by loso_grand_model_main.py),
print the statistics and build all figures. All functions live in loso_grand_model_functions.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# folder that holds loso_grand_model_functions.py
sys.path.append(r"C:\project\vsdi-face-decoding\scripts\functions_scripts")

from loso_grand_model_functions import (
    load_results,
    save_figure,
    print_summary,
    plot_heldout_accuracy,
    plot_inner_vs_heldout,
    plot_generalization_gap,
    plot_frame_vs_trial,
    plot_dataset_heatmap,
    plot_peak_window_groups,
    plot_fold_spread,
    plot_weight_maps,
    plot_group_weight_maps,
    plot_weight_norm_over_time,
    plot_weight_stability,
    plot_weight_pattern_similarity,
    plot_pixel_time_heatmap,
    print_group_check,
    plot_group_check_accuracy,
    plot_group_check_weights,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(r"C:\project\vsdi-face-decoding\results")
NPZ_NAME = "loso_nested_grand_model_results.npz"          # the full model
# NPZ_NAME = "loso_nested_grand_model_pilot.npz"          # <- use this line instead to look at a pilot run
CHECK_GROUPS = []             # e.g. ['face 2']: also make the single-group check figures from the FULL model
MAPS_TIMES_MS = [-100, 0, 50, 100, 150, 200, 300, 400]   # windows shown in the weight-map grid
ALPHA = 0.05
MIN_RUN = 3                                              # consecutive significant windows for "onset"

if __name__ == "__main__":
    npz_path = RESULTS_DIR / NPZ_NAME
    if not npz_path.exists():          # fall back to searching the project folder
        matches = sorted(Path(r"C:\project\vsdi-face-decoding").rglob(NPZ_NAME))
        if not matches:
            raise FileNotFoundError(f"{NPZ_NAME} not found in {RESULTS_DIR} or anywhere under the project folder")
        npz_path = matches[0]
        print(f"Not found in {RESULTS_DIR} - using {npz_path}")
    out_dir = npz_path.resolve().parent / "loso_nested_figures"

    results = load_results(npz_path)
    n_groups = len(results['group_names'])
    n_folds = results['ho_trial_acc'].shape[1]
    print(f"Loaded {npz_path}")
    print(f"  datasets: {len(results['dataset_ids'])}, left-out groups: {n_groups}, fold models per group: {n_folds}, "
          f"windows: {len(results['time_ms'])}, weights: {results['w_mean_windows'].shape}")
    print(f"  time range: {float(results['time_ms'][0]):.0f} to {float(results['time_ms'][-1]):.0f} ms")
    print(f"  parameters: {results['params']}\n")

    done = np.asarray(results['groups_done'], dtype=bool)
    names = results['group_names']
    check_groups = list(CHECK_GROUPS)

    if done.all():
        for metric in ('trial_acc', 'frame_acc'):
            print_summary(results, metric, alpha=ALPHA, min_run=MIN_RUN)
            print()

        # --- Accuracy ---
        fig, _ = plot_heldout_accuracy(results, 'trial_acc', alpha=ALPHA,
                                       title="Nested leave-one-face-out - left-out trial-level accuracy")
        save_figure(fig, out_dir, "01_heldout_accuracy_trial.png")

        fig, _ = plot_heldout_accuracy(results, 'frame_acc', alpha=ALPHA,
                                       title="Nested leave-one-face-out - left-out frame-level accuracy")
        save_figure(fig, out_dir, "02_heldout_accuracy_frame.png")

        save_figure(plot_inner_vs_heldout(results, 'trial_acc'), out_dir, "03_inner_vs_heldout.png")
        save_figure(plot_generalization_gap(results, 'trial_acc'), out_dir, "04_generalization_gap.png")
        save_figure(plot_frame_vs_trial(results), out_dir, "05_frame_vs_trial.png")
        save_figure(plot_dataset_heatmap(results, 'trial_acc'), out_dir, "06_dataset_heatmap.png")
        save_figure(plot_peak_window_groups(results, 'trial_acc'), out_dir, "07_peak_window_groups.png")
        save_figure(plot_fold_spread(results, 'trial_acc'), out_dir, "08_fold_spread.png")

        # --- Weights ---
        save_figure(plot_weight_maps(results, times_ms=MAPS_TIMES_MS), out_dir, "09_weight_maps.png")
        save_figure(plot_group_weight_maps(results), out_dir, "10_group_weight_maps_peak.png")
        save_figure(plot_weight_norm_over_time(results), out_dir, "11_weight_norm.png")
        save_figure(plot_weight_stability(results), out_dir, "12_weight_stability.png")
        save_figure(plot_weight_pattern_similarity(results), out_dir, "13_weight_pattern_similarity.png")
        save_figure(plot_pixel_time_heatmap(results), out_dir, "14_pixel_time_heatmap.png")

    else:
        check_groups = [n for n, d in zip(names, done) if d]
        print(f"PILOT / partial model: {int(done.sum())} of {len(done)} groups computed ({', '.join(check_groups)}).")
        print("Group-level statistics and figures need all groups - making the single-group check figures only.\n")

    for name in check_groups:
        tag = name.replace(' ', '_')
        print_group_check(results, name)
        print()
        save_figure(plot_group_check_accuracy(results, name), out_dir, f"check_{tag}_accuracy.png")
        save_figure(plot_group_check_weights(results, name), out_dir, f"check_{tag}_weights.png")

    print(f"Saved figures to {out_dir}")
    plt.show()