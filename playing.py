def run_groupkfold_cv(
    X_frames,
    y_frames,
    groups,
    n_splits=5,
    make_estimator=None,
    plot_splits=True,
    split_title="GroupKFold",
    verbose=False,
    return_models=False,
):
    """
    GroupKFold CV on frame-level samples with trial-level grouping.

    Returns a dict with:
      folds: list of per-fold info (indices + metrics)
      oof_pred: out-of-fold prediction for every sample
      confusion_matrix: confusion matrix over all OOF predictions
      acc_mean/std, bacc_mean/std
      cv: the GroupKFold object
      (optional) models: fitted model per fold

    Also plots the split visualization (like your screenshot) if plot_splits=True.
    """
    X = np.asarray(X_frames)
    y = np.asarray(y_frames).astype(int)
    g = np.asarray(groups)

    if X.ndim != 2:
        raise ValueError(f"X_frames must be 2D (n_samples, n_features). Got {X.shape}")
    if y.shape[0] != X.shape[0] or g.shape[0] != X.shape[0]:
        raise ValueError("X_frames, y_frames, and groups must have the same n_samples.")

    if make_estimator is None:
        def make_estimator():
            return Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LinearSVC(C=1.0, dual=True, max_iter=10000)),
            ])

    cv = GroupKFold(n_splits=n_splits)

    # Plot split diagram (like your example)
    if plot_splits:
        fig, ax = plt.subplots(figsize=(12, 3.2))
        plot_groupkfold_splits(X, y, g, cv, ax=ax, title=split_title)
        plt.tight_layout()
        plt.show()

    oof_pred = np.full(y.shape[0], -1, dtype=int)
    fold_infos = []
    models = [] if return_models else None

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, g)):
        est = make_estimator()
        est.fit(X[train_idx], y[train_idx])
        y_pred = est.predict(X[test_idx])
        oof_pred[test_idx] = y_pred

        acc = accuracy_score(y[test_idx], y_pred)
        bacc = balanced_accuracy_score(y[test_idx], y_pred)

        info = {
            "fold": fold,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "n_groups_train": int(np.unique(g[train_idx]).size),
            "n_groups_test": int(np.unique(g[test_idx]).size),
            "acc": float(acc),
            "balanced_acc": float(bacc),
        }
        fold_infos.append(info)

        if return_models:
            models.append(est)

        if verbose:
            print(
                f"Fold {fold}: acc={acc:.4f}, bacc={bacc:.4f} "
                f"(groups_test={info['n_groups_test']})"
            )

    if np.any(oof_pred < 0):
        raise RuntimeError("Some samples were never tested. Check groups / n_splits.")

    accs = np.array([f["acc"] for f in fold_infos], dtype=float)
    baccs = np.array([f["balanced_acc"] for f in fold_infos], dtype=float)

    results = {
        "cv": cv,
        "folds": fold_infos,
        "y_true": y,
        "groups": g,
        "oof_pred": oof_pred,
        "confusion_matrix": confusion_matrix(y, oof_pred),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "bacc_mean": float(baccs.mean()),
        "bacc_std": float(baccs.std(ddof=1)) if len(baccs) > 1 else 0.0,
    }
    if return_models:
        results["models"] = models

    return results



from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=5)

results = run_cv(
    X_frames, y_frames,
    groups=groups,
    splitter=cv,
    plot_splits=True,
    split_title="GroupKFold (frames-as-samples)",
    verbose=True,
    expect_full_coverage=True
)

print("\nSummary:")
print(f"Acc: {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
print(f"AUC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
print(results["confusion_matrix"])



def run_cv(X,y,groups=None,
        splitter=None,
        make_estimator=lambda: make_linear_svm(C_fixed),
        plot_splits=False,
        split_title="CV splits",
        expect_full_coverage=True,
        n_jobs=1,
        verbose=True):
    """
    Cross-validation runner with optional split plotting, OOF predictions (when full-coverage),
    and fold-wise + aggregated weight vectors for linear models.

    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    # groups is optional
    groups_arr = None
    if groups is not None:
        groups_arr = np.asarray(groups)

    if splitter is None:
        raise ValueError("splitter is required.")

    if make_estimator is None:
        def make_linear_svm(C):
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(C=C, dual=True, max_iter=10000))])

    # ---- Get splits ----
    if hasattr(splitter, "split"):
        splits = list(splitter.split(X, y, groups_arr))
        if plot_splits:
            if groups_arr is not None:
                fig, ax = plt.subplots(figsize=(12, 3.2))
                plot_groupkfold_splits(X, y, groups_arr, splitter, ax=ax, title=split_title)
                plt.tight_layout()
                plt.show()
            else:
                print("Note: plotting requires 'groups'. No plot was created.")
    else:
        # generator yielding (train_idx, test_idx)
        splits = list(splitter)
        if plot_splits:
            print(
                "Note: in this split method (e.g. leave-one-pair-out), plotting is not available "
                "because there are many overlapping test sets."
            )

    # ---- Full-coverage check (only when you request OOF outputs) ----
    if expect_full_coverage:
        counts = np.zeros(y.shape[0], dtype=int)
        for _, test_idx in splits:
            counts[np.asarray(test_idx)] += 1
        if not np.all(counts == 1):
            raise RuntimeError(
                "Not full-coverage: some samples are tested 0 or >1 times. "
                "Set expect_full_coverage=False for this splitter.")

    # Fold worker 
    def _fold_worker(fold_idx, train_idx, test_idx):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        est = make_estimator()
        est.fit(X[train_idx], y[train_idx])
        y_pred = est.predict(X[test_idx])
        acc = float(accuracy_score(y[test_idx], y_pred))
        scores = _get_scores_for_auc(est, X[test_idx])
        auc = _safe_auc(y[test_idx], scores)
        # ---- Extract weights if possible ----
        w_scaled = None
        w = None
        try:
            # Pipeline expected
            scaler = est.named_steps.get("scaler", None)
            clf = est.named_steps.get("clf", None)

            if clf is not None and hasattr(clf, "coef_"):
                w_scaled = np.asarray(clf.coef_).ravel()

                # back to original feature scale if scaler exists
                if scaler is not None and hasattr(scaler, "scale_"):
                    w = w_scaled / (np.asarray(scaler.scale_).ravel() + 1e-12)
                else:
                    w = w_scaled.copy()
        except Exception:
            # Keep None if anything goes wrong; don't crash CV run
            w_scaled = None
            w = None

        return {"fold": int(fold_idx),
                "test_idx": test_idx,
                "y_pred": y_pred,
                "scores": scores,
                "acc": acc,
                "auc": auc,
                "n_test": int(len(test_idx)),
                "w_scaled": w_scaled,
                "w": w}

    # ---- Run folds (optionally parallel) ----
    if n_jobs is None or n_jobs == 1:
        fold_results = [
            _fold_worker(i, train_idx, test_idx)
            for i, (train_idx, test_idx) in enumerate(splits)
        ]
    else:
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(_fold_worker)(i, train_idx, test_idx)
            for i, (train_idx, test_idx) in enumerate(splits)
        )

    # ---- Print per-fold summary ----
    if verbose:
        for f in fold_results:
            auc_txt = f"{f['auc']:.4f}" if np.isfinite(f["auc"]) else "nan"
            print(f"Fold {f['fold']}: acc={f['acc']:.4f}, AUC={auc_txt}, n_test={f['n_test']}")

    # ---- Summaries ----
    accs = np.array([f["acc"] for f in fold_results], dtype=float)
    aucs = np.array([f["auc"] for f in fold_results], dtype=float)

    results = {
        "folds": [
            {"fold": f["fold"], "acc": f["acc"], "auc": f["auc"], "n_test": f["n_test"]}
            for f in fold_results
        ],
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs, ddof=1)) if np.sum(np.isfinite(aucs)) > 1 else 0.0,
    }

    # ---- OOF predictions + confusion matrix (only valid for full-coverage CV) ----
    if expect_full_coverage:
        oof_pred = np.empty(y.shape[0], dtype=int)
        oof_scores = np.empty(y.shape[0], dtype=float)

        for f in fold_results:
            oof_pred[f["test_idx"]] = f["y_pred"]
            if f["scores"] is None:
                oof_scores[f["test_idx"]] = np.nan
            else:
                oof_scores[f["test_idx"]] = np.asarray(f["scores"], dtype=float)

        results["oof_pred"] = oof_pred
        results["oof_scores"] = oof_scores
        results["confusion_matrix"] = confusion_matrix(y, oof_pred)

    # ---- Aggregate weights across folds (if available) ----
    W = [f["w"] for f in fold_results if f["w"] is not None]
    if len(W) > 0:
        W = np.stack(W, axis=0)  # (n_folds, n_features)
        results["w_mean"] = W.mean(axis=0)
        results["w_std"] = W.std(axis=0, ddof=1) if W.shape[0] > 1 else np.zeros(W.shape[1], dtype=float)
    else:
        results["w_mean"] = None
        results["w_std"] = None

    # (Optional) also keep per-fold weights if you want later for stability plots
    results["fold_weights"] = [{"fold": f["fold"], "w": f["w"], "w_scaled": f["w_scaled"]}
            for f in fold_results]

    return results

