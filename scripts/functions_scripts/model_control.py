import numpy as np
from functions_scripts import ml_cv as cv


def run_permutation_nested_cv(X,y_trial,groups,
                            n_permutations=100,
                            outer_splitter=None,
                            inner_splitter=None,
                            C_grid=None,
                            metric="acc",
                            rule="one_se",
                            tie_break="smaller_C",
                            random_seed=42,
                            verbose=True):
    X = np.asarray(X)
    y_trial = np.asarray(y_trial).astype(int)
    groups = np.asarray(groups)

    if metric not in ("acc", "auc"):
        raise ValueError("metric must be 'acc' or 'auc'.")

    rng = np.random.default_rng(random_seed)
    unique_groups = np.unique(groups)

    # ensure alignment is possible
    if y_trial.size != unique_groups.size:
        raise ValueError(
            f"y_trial length ({y_trial.size}) must equal number of unique trials in groups ({unique_groups.size}). "
            "y_trial must be aligned with np.unique(groups) order."
        )

    shuffled_scores = np.empty(n_permutations, dtype=float)
    key = "outer_acc_mean" if metric == "acc" else "outer_auc_mean"
    progress_every = max(1, n_permutations // 10)

    for i in range(n_permutations):
        perm_trial_labels = y_trial.copy()
        rng.shuffle(perm_trial_labels)

        y_shuf = np.empty(groups.shape[0], dtype=int)
        for g, new_label in zip(unique_groups, perm_trial_labels):
            y_shuf[groups == g] = new_label

        perm_result = cv.run_nested_cv_selectC_then_eval(X, y_shuf, groups=groups,
                                                        outer_splitter=outer_splitter, inner_splitter=inner_splitter,
                                                        C_grid=C_grid,
                                                        metric=metric,
                                                        rule=rule,
                                                        tie_break=tie_break,
                                                        n_jobs_inner=1,
                                                        verbose=False)

        shuffled_scores[i] = float(perm_result[key])

        if verbose and ((i + 1) % progress_every == 0 or (i + 1) == n_permutations):
            print(f"  Permutation {i + 1}/{n_permutations} score={shuffled_scores[i]:.4f}")

    return {"shuffled_scores": shuffled_scores,
            "n_permutations": int(n_permutations),
            "random_seed": int(random_seed),
            "metric": metric,
            "chance_level": 0.5,
            "y_trial_order": unique_groups}







def permutation_significance_test(real_result, perm_result, metric=None,chance_level=0.5):
    """
    Compute summary stats + two-tailed p-value by 'distance from chance'.

    Inputs:
    real_result : dict
        Output of run_nested_cv_selectC_then_eval on REAL labels.
    perm_result : dict
        Output of run_permutation_nested_cv (shuffled labels).
    metric : str or None
        If None, taken from perm_result['metric'].
    chance_level : float or None
        If None, taken from perm_result['chance_level'] (default 0.5).

    output:
    dict with:
    - real_score
    - shuffled_scores
    - shuffled_mean/std/min/max
    - p_value_two_tailed
    - metric
    - chance_level
    """
    if metric not in ("acc", "auc"):
        raise ValueError("metric must be 'acc' or 'auc'.")

    key = "outer_acc_mean" if metric == "acc" else "outer_auc_mean"
    real_score = float(real_result[key])

    shuffled_scores = np.asarray(perm_result["shuffled_scores"], dtype=float)
    if shuffled_scores.ndim != 1 or shuffled_scores.size == 0:
        raise ValueError("perm_result['shuffled_scores'] must be a non-empty 1D array.")

    # two-tailed: compare distance-from-chance
    p_value = float(np.mean(np.abs(shuffled_scores - chance_level) >= np.abs(real_score - chance_level)))

    shuffled_mean = float(np.mean(shuffled_scores))
    shuffled_std = float(np.std(shuffled_scores, ddof=1)) if shuffled_scores.size > 1 else float("nan")
    shuffled_min = float(np.min(shuffled_scores))
    shuffled_max = float(np.max(shuffled_scores))

    return {
        "metric": metric,
        "chance_level": float(chance_level),
        "real_score": real_score,
        "shuffled_scores": shuffled_scores,
        "shuffled_mean": shuffled_mean,
        "shuffled_std": shuffled_std,
        "shuffled_min": shuffled_min,
        "shuffled_max": shuffled_max,
        "p_value_two_tailed": p_value,
        "pass_alpha_0p05": bool(p_value < 0.05),
    }
