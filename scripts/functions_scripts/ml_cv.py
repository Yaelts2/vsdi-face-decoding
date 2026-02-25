# ml_cv.py
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


# -----------------------------
# Model factories
# -----------------------------
def make_linear_svm(C: float = 0.001, max_iter: int = 10000) -> LinearSVC:
    return LinearSVC(C=C, dual=True, max_iter=max_iter)


# -----------------------------
# Scoring utilities
# -----------------------------
def compute_auc(estimator, X_test, y_true) -> Tuple[float, Optional[np.ndarray]]:
    """
    Compute ROC-AUC for binary classification with best-effort score extraction.
    Returns (auc, scores). If AUC can't be computed -> (np.nan, None).
    """
    y_true = np.asarray(y_true).astype(int)

    if np.unique(y_true).size < 2:
        return np.nan, None

    scores = None

    if hasattr(estimator, "decision_function"):
        s = np.asarray(estimator.decision_function(X_test))
        if s.ndim == 1:
            scores = s
        elif s.ndim == 2 and s.shape[1] >= 2:
            scores = s[:, 1]

    if scores is None and hasattr(estimator, "predict_proba"):
        p = np.asarray(estimator.predict_proba(X_test))
        if p.ndim == 2 and p.shape[1] >= 2:
            scores = p[:, 1]

    if scores is None:
        return np.nan, None

    try:
        return float(roc_auc_score(y_true, scores)), scores
    except Exception:
        return np.nan, scores


# -----------------------------
# Split utilities
# -----------------------------
def get_splits(splitter, X, y, groups=None):
    """
    Normalize a splitter into a list of (train_idx, test_idx).
    Supports:
    - sklearn splitters (have .split)
    - custom splitter functions: splitter(y, groups)
    """
    y = np.asarray(y)
    n_samples = y.shape[0]

    # --- Case 1: sklearn-style splitter object ---
    if hasattr(splitter, "split"):
        if groups is None:
            it = splitter.split(X, y)
        else:
            it = splitter.split(X, y, np.asarray(groups))

    # --- Case 2: custom splitter function ---
    elif callable(splitter):
        if groups is None:
            raise ValueError("Custom splitter function requires 'groups'.")
        it = splitter(y, groups)

    # --- Case 3: already an iterable of splits ---
    else:
        it = splitter

    splits = []
    for tr, te in it:
        tr = np.asarray(tr, dtype=int).ravel()
        te = np.asarray(te, dtype=int).ravel()

        if tr.size == 0 or te.size == 0:
            raise ValueError("Encountered an empty train or test split.")
        if tr.min() < 0 or te.min() < 0 or tr.max() >= n_samples or te.max() >= n_samples:
            raise ValueError("Split indices out of bounds.")

        splits.append((tr, te))

    if len(splits) == 0:
        raise ValueError("Splitter produced 0 splits.")

    return splits


def has_full_test_coverage(splits: Sequence[Tuple[np.ndarray, np.ndarray]], n_samples: int) -> bool:
    """
    True iff each sample appears in exactly one test fold.
    Needed for OOF predictions / confusion matrix / ROC.
    """
    test_counts = np.zeros(n_samples, dtype=int)
    for _, te in splits:
        te = np.asarray(te, dtype=int)
        test_counts[te] += 1
    return bool(np.all(test_counts == 1))


# -----------------------------
# Linear weights (for maps)
# -----------------------------
def extract_linear_weights_general(estimator):
    """
    Robust weight extraction:
    - If estimator is a Pipeline, tries named_steps["clf"]
    - Otherwise uses estimator.coef_
    Returns 1D weight vector (n_features,)
    """
    # Pipeline case
    if hasattr(estimator, "named_steps"):
        if "clf" in estimator.named_steps and hasattr(estimator.named_steps["clf"], "coef_"):
            return np.asarray(estimator.named_steps["clf"].coef_).ravel()
        # fallback: find any step with coef_
        for step in estimator.named_steps.values():
            if hasattr(step, "coef_"):
                return np.asarray(step.coef_).ravel()
        return None
    # Plain estimator
    if hasattr(estimator, "coef_"):
        return np.asarray(estimator.coef_).ravel()
    return None


# ------------------------------
# Trial-level aggregation (majority vote) from frame-level predictions
# ------------------------------
def majority_vote_trial_predictions(y_pred, y_true, groups):
    # 1. Get trial IDs and an index mapping every frame to its trial
    trial_ids, inverse = np.unique(groups, return_inverse=True)
    n_trials = trial_ids.size
    
    y_pred_trials = np.empty(n_trials, dtype=int)
    y_true_trials = np.empty(n_trials, dtype=int)

    # 2. Iterate using the pre-calculated inverse indices
    for i in range(n_trials):
        idx = (inverse == i)
        # Take the first true label (assumes trial labels are consistent)
        y_true_trials[i] = y_true[idx][0]
        # Majority vote (works for binary and multiclass)
        y_pred_trials[i] = np.bincount(y_pred[idx]).argmax()

    return y_true_trials, y_pred_trials


# -----------------------------
# Fold execution
# -----------------------------

def fit_eval_one_fold(
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    make_estimator: Callable[[], Any],
) -> Dict[str, Any]:
    """Fit on train split and evaluate on test split; return fold artifacts."""
    train_idx = np.asarray(train_idx, dtype=int).ravel()
    test_idx = np.asarray(test_idx, dtype=int).ravel()

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    est = make_estimator()
    est.fit(X_tr, y_tr)

    y_pred = est.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))

    auc, scores = compute_auc(est, X_te, y_te)

    w = extract_linear_weights_general(est)

    return {
        "fold": int(fold_idx),
        "test_idx": test_idx,
        "y_pred": np.asarray(y_pred, dtype=int),
        "scores": None if scores is None else np.asarray(scores, dtype=float),
        "acc": acc,
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "n_test": int(test_idx.size),
        "w": w,
    }


def summarize_folds(fold_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate fold metrics; optionally aggregate weights if present."""
    if len(fold_results) == 0:
        raise ValueError("fold_results is empty")

    accs = np.asarray([f["acc"] for f in fold_results], dtype=float)
    aucs = np.asarray([f["auc"] for f in fold_results], dtype=float)

    out: Dict[str, Any] = {
        "folds": [
            {"fold": int(f["fold"]), "acc": float(f["acc"]), "auc": float(f["auc"]), "n_test": int(f["n_test"])}
            for f in fold_results
        ],
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)) if accs.size > 1 else np.nan,
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs, ddof=1)) if np.sum(np.isfinite(aucs)) > 1 else np.nan,
    }

    W = [f["w"] for f in fold_results if f.get("w") is not None]
    if len(W) > 0:
        W = np.stack(W, axis=0)
        out["w_mean"] = W.mean(axis=0)
        out["w_std"] = W.std(axis=0, ddof=1) if W.shape[0] > 1 else np.full(W.shape[1], np.nan)
    else:
        out["w_mean"] = None
        out["w_std"] = None

    return out


def add_oof_outputs(results: Dict[str, Any], fold_results: Sequence[Dict[str, Any]], y: np.ndarray) -> Dict[str, Any]:
    """Create OOF vectors (pred, scores) + confusion matrix (requires full coverage)."""
    y = np.asarray(y).astype(int)
    n = y.shape[0]

    oof_pred = np.full(n, fill_value=-1, dtype=int)
    oof_scores = np.full(n, fill_value=np.nan, dtype=float)

    for f in fold_results:
        te = np.asarray(f["test_idx"], dtype=int)
        oof_pred[te] = np.asarray(f["y_pred"], dtype=int)

        scores = f.get("scores", None)
        if scores is not None:
            oof_scores[te] = np.asarray(scores, dtype=float)

    if np.any(oof_pred < 0):
        missing = int(np.sum(oof_pred < 0))
        raise RuntimeError(
            f"OOF aggregation failed: {missing} samples never appeared in a test fold."
        )

    results["oof_y_true"] = y
    results["oof_pred"] = oof_pred
    results["oof_scores"] = oof_scores
    results["oof_has_score"] = np.isfinite(oof_scores)
    results["confusion_matrix"] = confusion_matrix(y, oof_pred)
    return results



# -----------------------------
# split LOPO
# -----------------------------
def leave_one_group_pair_out_splits(y, groups, seed=0):
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    all_idx = np.arange(len(y))

    # group → label (must be pure)
    g2lab = {}
    for g in np.unique(groups):
        labs = np.unique(y[groups == g])
        if labs.size != 1:
            raise ValueError("Groups must be label-pure")
        g2lab[g] = int(labs[0])

    pos_g = [g for g, l in g2lab.items() if l == 1]
    neg_g = [g for g, l in g2lab.items() if l == 0]

    rng = np.random.default_rng(seed)
    rng.shuffle(pos_g)
    rng.shuffle(neg_g)

    m = min(len(pos_g), len(neg_g))
    for gp, gn in zip(pos_g[:m], neg_g[:m]):
        test = (groups == gp) | (groups == gn)
        yield all_idx[~test], all_idx[test]
        
        
        
# -----------------------------
# Main CV runner
# -----------------------------

def run_cv(X, y,splitter,groups=None,
        make_estimator: Optional[Callable[[], Any]] = None,
        expect_full_coverage: bool = True,
        n_jobs: int = 1,
        verbose: bool = True,
        keep_fold_artifacts: bool = False) -> Dict[str, Any]:
    """
    Core CV runner (single-layer CV).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups_arr = None if groups is None else np.asarray(groups)

    if make_estimator is None:
        make_estimator = lambda: make_linear_svm(C=0.001)

    splits = get_splits(splitter, X, y, groups_arr)

    if expect_full_coverage and not has_full_test_coverage(splits, n_samples=y.shape[0]):
        raise RuntimeError(
            "Splitter is not full-coverage (some samples tested 0 or >1 times). "
            "Set expect_full_coverage=False or use a different splitter."
        )

    if n_jobs is None or n_jobs == 1:
        fold_results = [
            fit_eval_one_fold(i, X, y, tr, te, make_estimator)
            for i, (tr, te) in enumerate(splits)
        ]
    else:
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(fit_eval_one_fold)(i, X, y, tr, te, make_estimator)
            for i, (tr, te) in enumerate(splits)
        )
    if verbose:
        for f in fold_results:
            auc_txt = f"{f['auc']:.4f}" if np.isfinite(f["auc"]) else "nan"
            print(f"Fold {f['fold']}: acc={f['acc']:.4f}, auc={auc_txt}, n_test={f['n_test']}")

    results = summarize_folds(fold_results)

    if expect_full_coverage:
        results = add_oof_outputs(results, fold_results, y)

    if keep_fold_artifacts:
        results["fold_results"] = fold_results
        results["splits"] = splits

    return results


# -----------------------------
# Hyperparameter selection (C)
# -----------------------------

def select_best_C(X, y, splitter,
                make_estimator_for_C: Callable[[float], Any],
                C_grid=None,
                groups=None,
                metric: str = "acc",
                tie_break: str = "smaller_C",
                rule: str = "one_se",               # recommended default for robustness
                expect_full_coverage: bool = False,
                n_jobs: int = 1,
                verbose: bool = True):
    """
    Select C by running run_cv for each C and choosing best by metric.
    Returns (best_C, rows, info).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups_arr = None if groups is None else np.asarray(groups)

    if C_grid is None:
        C_grid = np.logspace(-4, 2, 7)
    C_grid = np.asarray(list(C_grid), dtype=float)
    if C_grid.size == 0:
        raise ValueError("C_grid is empty")

    if metric not in ("acc", "auc"):
        raise ValueError("metric must be 'acc' or 'auc'")
    if tie_break not in ("smaller_C", "larger_C"):
        raise ValueError("tie_break must be 'smaller_C' or 'larger_C'")
    if rule not in ("best", "one_se"):
        raise ValueError("rule must be 'best' or 'one_se'")

    key = "acc_mean" if metric == "acc" else "auc_mean"
    std_key = "acc_std" if metric == "acc" else "auc_std"

    rows = []
    for C in C_grid:
        res = run_cv(X, y,splitter=splitter,groups=groups_arr,
                    make_estimator=lambda C=C: make_estimator_for_C(C),
                    expect_full_coverage=expect_full_coverage,
                    n_jobs=n_jobs,
                    verbose=False)
        rows.append({
            "C": float(C),
            "acc_mean": float(res["acc_mean"]),
            "acc_std": float(res["acc_std"]),
            "auc_mean": float(res["auc_mean"]),
            "auc_std": float(res["auc_std"]),
            "n_folds": int(len(res["folds"])),
        })

    scores = np.array([r[key] for r in rows], dtype=float)
    best_idx = int(np.nanargmax(scores))
    best_score = float(scores[best_idx])

    n_folds = np.array([max(r["n_folds"], 1) for r in rows], dtype=float)
    se = np.array([r[std_key] / np.sqrt(n) for r, n in zip(rows, n_folds)], dtype=float)

    if rule == "one_se":
        thresh = best_score - float(se[best_idx]) if np.isfinite(se[best_idx]) else best_score
        candidates = [i for i, s in enumerate(scores) if np.isfinite(s) and s >= thresh]
    else:
        tol = float(np.nanmin(se[np.isfinite(se)])) if np.any(np.isfinite(se)) else 0.0
        candidates = [i for i, s in enumerate(scores) if np.isfinite(s) and s >= best_score - tol]

    chosen_idx = (
        min(candidates, key=lambda i: rows[i]["C"])
        if tie_break == "smaller_C"
        else max(candidates, key=lambda i: rows[i]["C"])
    )

    best_C = rows[chosen_idx]["C"]

    if verbose:
        print(f"Selected C={best_C:g} using metric={metric}, rule={rule}, tie_break={tie_break}.")
        for r in rows:
            print(f"C={r['C']:.4g} | acc={r['acc_mean']:.4f}±{r['acc_std']:.4f} | "
                f"auc={r['auc_mean']:.4f}±{r['auc_std']:.4f}")

    info = {
        "metric": metric,
        "rule": rule,
        "tie_break": tie_break,
        "key": key,
        "best_score": best_score,
        "best_idx_raw": best_idx,
        "chosen_idx": chosen_idx,
        "candidates": candidates,
    }
    return best_C, rows, info


# -----------------------------
# Nested CV
# -----------------------------

def choose_final_C(chosen_Cs: Sequence[float], method: str = "mode_then_median") -> float:
    """Pick a single C for the final model from per-outer-fold chosen Cs."""
    chosen_Cs = np.asarray(chosen_Cs, dtype=float)
    if chosen_Cs.size == 0:
        raise ValueError("chosen_Cs is empty")
    if method == "median":
        return float(np.median(chosen_Cs))
    vals, counts = np.unique(chosen_Cs, return_counts=True)
    if method == "mode":
        return float(vals[np.argmax(counts)])
    if method == "mode_then_median":
        maxc = np.max(counts)
        modes = vals[counts == maxc]
        return float(np.median(modes))
    raise ValueError("method must be 'median', 'mode', or 'mode_then_median'")



def run_nested_cv_selectC_then_eval(X, y, groups, outer_splitter, inner_splitter,
                                    C_grid=None,
                                    metric: str = "acc", tie_break: str = "smaller_C", rule: str = "one_se",
                                    n_jobs_inner: int = 1,
                                    verbose: bool = True,
                                    ) -> Dict[str, Any]:
    """
    Nested CV:
    Outer: evaluate generalization (GroupKFold on trials)
    Inner: select C using only outer-train data
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    outer_splits = get_splits(outer_splitter, X, y, groups)

    # OOF containers 
    n = y.shape[0]
    oof_pred = np.full(n, -1, dtype=int)
    oof_scores = np.full(n, np.nan, dtype=float)

    #trial-level OOF containers 
    oof_trial_true: List[int] = []
    oof_trial_pred: List[int] = []

    outer_folds: List[Dict[str, Any]] = []
    chosen_Cs: List[float] = []
    fold_weights: List[np.ndarray] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits):
        X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], groups[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # nest C on outer-train (inner splitter)
        best_C, rows, _info = select_best_C(X_tr, y_tr,
                                            splitter=inner_splitter,
                                            groups=g_tr,
                                            C_grid=C_grid,
                                            metric=metric,
                                            tie_break=tie_break,
                                            rule=rule,
                                            expect_full_coverage=False,
                                            n_jobs=n_jobs_inner,
                                            verbose=False,
                                            make_estimator_for_C=lambda C: make_linear_svm(C))
        chosen_Cs.append(float(best_C))

        clf = make_linear_svm(best_C)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        acc = float(np.mean(y_pred == y_te))
        auc, scores = compute_auc(clf, X_te, y_te)

        # accuracy across trials (majority vote across frames)
        g_te = groups[te_idx]
        y_true_trial, y_pred_trial = majority_vote_trial_predictions(y_pred, y_te, g_te)
        acc_trial = float(np.mean(y_pred_trial == y_true_trial))

        #accumulate trial-level OOF arrays across folds
        oof_trial_true.extend(np.asarray(y_true_trial, dtype=int).ravel().tolist())
        oof_trial_pred.extend(np.asarray(y_pred_trial, dtype=int).ravel().tolist())
    
        # Fill OOF (frame-level)
        te_idx = np.asarray(te_idx, dtype=int)
        oof_pred[te_idx] = y_pred.astype(int)
        if scores is not None:
            oof_scores[te_idx] = np.asarray(scores, dtype=float)

        # Weights (only for linear models)
        w = extract_linear_weights_general(clf)
        if w is not None:
            fold_weights.append(w)

        outer_folds.append({
            "outer_fold": int(fold_idx),
            "best_C": float(best_C),
            "acc": float(acc),
            "auc": float(auc) if np.isfinite(auc) else np.nan,
            "acc_trial": float(acc_trial),
            "n_test": int(len(te_idx)),
        })

        if verbose:
            auc_txt = f"{auc:.4f}" if np.isfinite(auc) else "nan"
            print(f"[Outer {fold_idx}] C={best_C:g} | acc={acc:.4f} | auc={auc_txt} | acc_trial={acc_trial:.4f} | n_test={len(te_idx)}")

    if np.any(oof_pred < 0):
        raise RuntimeError("Outer OOF aggregation failed: some samples never tested.")

    accs = np.asarray([f["acc"] for f in outer_folds], dtype=float)
    aucs = np.asarray([f["auc"] for f in outer_folds], dtype=float)
    acc_trials = np.asarray([f["acc_trial"] for f in outer_folds], dtype=float)

    final_C = choose_final_C(chosen_Cs, method="mode_then_median")

    out: Dict[str, Any] = {
        "outer_folds": outer_folds,
        "outer_acc_mean": float(np.mean(accs)),
        "outer_acc_std": float(np.std(accs, ddof=1)) if accs.size > 1 else np.nan,
        "outer_auc_mean": float(np.nanmean(aucs)),
        "outer_auc_std": float(np.nanstd(aucs, ddof=1)) if np.sum(np.isfinite(aucs)) > 1 else np.nan,

        "outer_acc_trial_mean": float(np.mean(acc_trials)),
        "outer_acc_trial_std": float(np.std(acc_trials, ddof=1)) if acc_trials.size > 1 else np.nan,

        "chosen_Cs": chosen_Cs,
        "final_C": float(final_C),

        # OOF (frame-level)
        "oof_y_true": y,
        "oof_pred": oof_pred,
        "oof_scores": oof_scores,
        "oof_has_score": np.isfinite(oof_scores),
        "confusion_matrix": confusion_matrix(y, oof_pred),
        
    }

    #trial-level OOF arrays + confusion matrix
    out["oof_y_true_trial"] = np.asarray(oof_trial_true, dtype=int)
    out["oof_pred_trial"] = np.asarray(oof_trial_pred, dtype=int)
    out["confusion_matrix_trial"] = confusion_matrix(out["oof_y_true_trial"], out["oof_pred_trial"])

    # mean fold weights (stability map)
    if len(fold_weights) > 0:
        W = np.stack(fold_weights, axis=0)   # (n_folds, n_roi_pixels)
        out["W_outer"] = W
        out["w_outer_mean"] = W.mean(axis=0)
        out["w_outer_std"] = W.std(axis=0, ddof=1) if W.shape[0] > 1 else np.full(W.shape[1], np.nan)
    else:
        out["W_outer"] = None
        out["w_outer_mean"] = None
        out["w_outer_std"] = None

    return out




def fit_final_model(X, y, C_final: float) -> Dict[str, Any]:
    """
    Fit the final linear SVM on all data using C_final and return model + weights.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    est = make_linear_svm(C_final)
    est.fit(X, y)
    w = extract_linear_weights_general(est)
    return {"estimator": est, "C": float(C_final), "w_scaled": w_scaled, "w": w}
