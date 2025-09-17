"""
Multiple Linear Regression via numpy (OLS).
Target: total_value; Features: items_count, has_promo, avg_item_price, marketing_channel dummies.
"""
from .io_utils import load_csv
import random


def _float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _one_hot(value, categories):
    return [1.0 if value == c else 0.0 for c in categories]


def _train_test_split(X, y, test_size=0.2, seed=7):
    rnd = random.Random(seed)
    idx = list(range(len(y)))
    rnd.shuffle(idx)
    n_test = max(1, int(len(y) * test_size))
    test_idx = set(idx[:n_test])
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in range(len(y)):
        if i in test_idx:
            Xte.append(X[i]); yte.append(y[i])
        else:
            Xtr.append(X[i]); ytr.append(y[i])
    return Xtr, ytr, Xte, yte


def _to_numpy(mat):
    try:
        import numpy as np
        return np.array(mat, dtype=float)
    except Exception:
        return None


def _ols_fit(X, y):
    np_X = _to_numpy(X)
    np_y = _to_numpy(y)
    if np_X is None or np_y is None:
        raise RuntimeError("NumPy is required for OLS. Please install numpy.")
    import numpy as np
    beta, *_ = np.linalg.lstsq(np_X, np_y, rcond=None)
    return beta


def _predict(X, beta):
    import numpy as np
    return np.array(X) @ beta


def _rmse(y_true, y_pred):
    import numpy as np
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def _r2(y_true, y_pred):
    import numpy as np
    yt = np.array(y_true); yp = np.array(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def run_regression(transactions_csv: str):
    records, _ = load_csv(transactions_csv)
    channels = sorted({r.get("marketing_channel", "NA") for r in records})
    X, y = [], []
    for r in records:
        items_count = _float(r.get("items_count", 0))
        total_value = _float(r.get("total_value", 0))
        has_promo = 1.0 if str(r.get("has_promo", "0")).strip() in ("1", "True", "true", "yes") else 0.0
        avg_price = total_value / items_count if items_count > 0 else 0.0
        ch = r.get("marketing_channel", "NA")
        row = [1.0, items_count, has_promo, avg_price] + _one_hot(ch, channels)
        X.append(row); y.append(total_value)
    Xtr, ytr, Xte, yte = _train_test_split(X, y, test_size=0.2, seed=11)
    beta = _ols_fit(Xtr, ytr)
    yhat_tr = _predict(Xtr, beta)
    yhat_te = _predict(Xte, beta)
    metrics = {
        "rmse_train": _rmse(ytr, yhat_tr),
        "rmse_test": _rmse(yte, yhat_te),
        "r2_train": _r2(ytr, yhat_tr),
        "r2_test": _r2(yte, yhat_te),
        "n_train": len(ytr),
        "n_test": len(yte),
        "feature_order": ["bias", "items_count", "has_promo", "avg_item_price"] + [f"channel={c}" for c in channels]
    }
    return beta.tolist(), metrics