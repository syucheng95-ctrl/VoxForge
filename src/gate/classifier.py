"""Gate classifier: model loading and prediction logic (no file I/O)."""

import numpy as np
import joblib
import sklearn

sklearn.set_config(transform_output="default")


def micro_dice(t, p, g):
    return 2.0 * t / max(1.0, p + g)


def load_gate_model(model_path):
    """Load gate model bundle from joblib file.
    Returns (preprocessor, model).
    """
    bundle = joblib.load(model_path)
    return bundle["preprocessor"], bundle["model"]


def predict_gate(model, preprocessor, rows, kept_features, threshold=0.5):
    """Run gate inference on a list of dict rows.

    Args:
        model: sklearn classifier with predict_proba
        preprocessor: sklearn ColumnTransformer
        rows: list of dicts (feature columns)
        kept_features: list of feature column names to use
        threshold: decision threshold (default 0.5)

    Returns:
        use_s2: np.array of int (0 or 1)
        p_use_s2: np.array of float (probability of class 1)
    """
    cat_in = [
        c
        for c in kept_features
        if c
        in [
            "feature__pred_category",
            "feature__anatomy_target",
            "feature__laterality",
            "feature__anatomy_group",
            "feature__final_policy",
            "feature__final_tightness",
            "feature__expert",
        ]
    ]
    num_in = [c for c in kept_features if c not in cat_in]
    nc = len(cat_in)
    n_num = len(num_in)

    X = np.empty((len(rows), nc + n_num), dtype=object)
    for i in range(nc):
        X[:, i] = [r.get(cat_in[i], "") for r in rows]
    for j in range(n_num):
        X[:, nc + j] = np.array(
            [float(r.get(num_in[j], 0) or 0) for r in rows], dtype=np.float64
        )

    X_proc = preprocessor.transform(X)
    proba = model.predict_proba(np.asarray(X_proc))
    if proba.shape[1] == 1:
        p_use_s2 = np.zeros(len(rows))
    else:
        p_use_s2 = proba[:, 1]
    use_s2 = (p_use_s2 >= threshold).astype(int)

    return use_s2, p_use_s2
