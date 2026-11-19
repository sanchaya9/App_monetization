"""Core ML logic for the App Monetization dashboard (no Streamlit imports here,
so it can be tested and reused from scripts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

try:  # real XGBoost if installed, otherwise a scikit-learn stand-in
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

RANDOM_STATE = 42
REVENUE_HINTS = ("revenue", "gross", "earning", "income", "sales", "mrr", "arpu", "ltv")
TARGET_HINTS = ("top", "grossing", "label", "target", "class")
ID_HINTS = ("id", "name", "title", "url", "link", "package", "bundle", "developer", "description")


# ----------------------------------------------------------------------------
# Demo data (synthetic - clearly NOT the Kaggle data)
# ----------------------------------------------------------------------------
def make_demo_data(n: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(["Games", "Finance", "Social", "Health", "Education", "Tools"], n)
    price = rng.choice([0, 0.99, 1.99, 2.99, 4.99, 9.99], n)
    ret = np.clip(rng.beta(2, 6, n), 0, 1)
    ad = np.clip(rng.gamma(2, 2, n), 0, 15)
    rating = np.clip(rng.normal(4.0, 0.7, n), 1, 5)
    iap = rng.integers(0, 2, n)
    z = -5.5 + 7 * ret + 0.5 * (rating - 4) + 0.9 * iap - 0.12 * ad + 0.8 * (cat == "Games")
    p = 1 / (1 + np.exp(-z))
    return pd.DataFrame({
        "app_id": np.arange(n), "category": cat, "price": price,
        "day7_retention": ret.round(3), "ad_load": ad.round(3),
        "rating": rating.round(2), "has_iap": iap,
        "is_top_grossing": (rng.random(n) < p).astype(int),
    })


# ----------------------------------------------------------------------------
# Column handling
# ----------------------------------------------------------------------------
def _is_binary_like(s: pd.Series) -> bool:
    vals = set(s.dropna().astype(str).str.strip().str.lower().unique())
    return 1 < len(vals) <= 2 and vals <= {"0", "1", "yes", "no", "true", "false", "y", "n", "0.0", "1.0"}


def guess_target(df: pd.DataFrame) -> str:
    """Prefer a binary 'top grossing'-style column, else a revenue-like numeric column."""
    for c in df.columns:
        if any(h in c.lower() for h in TARGET_HINTS) and _is_binary_like(df[c]):
            return c
    for c in df.columns:
        if _is_binary_like(df[c]) and any(h in c.lower() for h in ("gross", "revenue", "top")):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and any(h in c.lower() for h in REVENUE_HINTS):
            return c
    return df.columns[-1]


def guess_id_columns(df: pd.DataFrame, target: str) -> list[str]:
    out = []
    for c in df.columns:
        if c == target:
            continue
        low = c.lower()
        looks_id = low in ("id", "app_id") or low.endswith("_id") or any(h == low or low.endswith("_" + h) for h in ID_HINTS)
        near_unique = df[c].nunique(dropna=True) > 0.9 * len(df) and not pd.api.types.is_float_dtype(df[c])
        if looks_id or near_unique:
            out.append(c)
    return out


def guess_leak_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Revenue-like columns other than the target: they would leak the answer."""
    return [c for c in df.columns if c != target and any(h in c.lower() for h in REVENUE_HINTS)]


def build_label(df: pd.DataFrame, target: str, top_pct: float = 10.0) -> tuple[pd.Series, str]:
    """Return (0/1 label, description). Revenue columns become 'top X%' flags."""
    s = df[target]
    if _is_binary_like(s):
        m = s.astype(str).str.strip().str.lower().map(
            {"1": 1, "1.0": 1, "yes": 1, "true": 1, "y": 1, "0": 0, "0.0": 0, "no": 0, "false": 0, "n": 0})
        return m.astype(int), f"'{target}' used as the 0/1 top-grossing flag"
    s = pd.to_numeric(s, errors="coerce")
    cut = s.quantile(1 - top_pct / 100)
    return (s >= cut).astype(int), f"'{target}' converted to a label: top {top_pct:.0f}% by value (>= {cut:,.2f}) = top-grossing"


def prepare(df: pd.DataFrame, target: str, drop_cols: list[str], top_pct: float = 10.0):
    y, desc = build_label(df, target, top_pct)
    ok = y.notna()
    X = df.loc[ok].drop(columns=[c for c in set(drop_cols + [target]) if c in df.columns]).copy()
    # dates / free text with huge cardinality are not useful features
    for c in list(X.columns):
        if X[c].dtype == object and X[c].nunique() > 50:
            X = X.drop(columns=c)
    return X, y.loc[ok].astype(int), desc


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat),
    ])


def _models(pos_weight: float):
    models = {
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                          {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [5, 20, 50]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=RANDOM_STATE),
                          {"clf__max_depth": [6, 10, None], "clf__min_samples_leaf": [1, 5, 10]}),
    }
    if HAS_XGB:
        models["XGBoost"] = (XGBClassifier(n_estimators=250, learning_rate=0.08, max_depth=4, subsample=0.9,
                                           colsample_bytree=0.9, scale_pos_weight=pos_weight,
                                           eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
                             {"clf__max_depth": [3, 4, 6], "clf__learning_rate": [0.05, 0.1]})
    else:
        models["Gradient Boosting (XGBoost stand-in)"] = (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"clf__max_depth": [3, 5, None], "clf__learning_rate": [0.05, 0.1]})
    return models


# ----------------------------------------------------------------------------
# Metrics / thresholds
# ----------------------------------------------------------------------------
def metrics_at(y_true, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "threshold": round(float(thr), 3),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f))]) if len(t) else 0.5


def threshold_sweep(y_true, proba, cost_per_app: float, value_per_hit: float, budget: float | None = None) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    rows = []
    for thr in np.linspace(0.01, 0.99, 99):
        pred = proba >= thr
        backed = int(pred.sum())
        hits = int((pred & (y_true == 1)).sum())
        spend = backed * cost_per_app
        rows.append({
            "threshold": round(float(thr), 2), "apps_backed": backed, "hits_captured": hits,
            "wasted": backed - hits, "spend": spend,
            "precision": hits / backed if backed else 0.0,
            "recall": hits / max(1, int((y_true == 1).sum())),
            "net_value": hits * value_per_hit - spend,
            "within_budget": (budget is None) or spend <= budget,
        })
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame) -> pd.Series:
    ok = sweep[sweep["within_budget"]]
    if ok.empty:  # budget too small for any selection: fall back to the cheapest option
        return sweep.sort_values("apps_backed").iloc[0]
    return ok.sort_values(["net_value", "threshold"], ascending=[False, False]).iloc[0]


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def run_models(X: pd.DataFrame, y: pd.Series, tune: bool = False, test_size: float = 0.2) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    pos_weight = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    results, fitted = {}, {}
    for name, (est, grid) in _models(pos_weight).items():
        pipe = Pipeline([("prep", _preprocessor(X)), ("clf", est)])
        best_params = {}
        if tune:
            cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
            search = RandomizedSearchCV(pipe, grid, n_iter=min(6, int(np.prod([len(v) for v in grid.values()]))),
                                        scoring="average_precision", cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
            search.fit(Xtr, ytr)
            pipe, best_params = search.best_estimator_, search.best_params_
        else:
            pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        pr_p, pr_r, _ = precision_recall_curve(yte, proba)
        results[name] = {
            "proba": proba, "roc_auc": roc_auc_score(yte, proba), "pr_auc": average_precision_score(yte, proba),
            "roc_curve": (fpr, tpr), "pr_curve": (pr_p, pr_r),
            "best_f1_thr": best_f1_threshold(yte, proba), "best_params": best_params,
        }
        fitted[name] = pipe
    champion = max(results, key=lambda k: results[k]["pr_auc"])
    return {"results": results, "models": fitted, "champion": champion,
            "X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte,
            "feature_names": list(X.columns)}


def metrics_table(out: dict) -> pd.DataFrame:
    rows = []
    for name, r in out["results"].items():
        for label, thr in (("default 0.50", 0.5), (f"best-F1 ({r['best_f1_thr']:.2f})", r["best_f1_thr"])):
            m = metrics_at(out["y_test"], r["proba"], thr)
            rows.append({"model": name, "setting": label, "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                         **{k: m[k] for k in ("accuracy", "precision", "recall", "f1")}})
    return pd.DataFrame(rows)


def confusion_df(y_true, proba, thr: float) -> pd.DataFrame:
    cm = confusion_matrix(y_true, (proba >= thr).astype(int), labels=[0, 1])
    return pd.DataFrame(cm, index=["actual_not_top", "actual_top"], columns=["pred_not_top", "pred_top"])


def feature_importance(out: dict, n_repeats: int = 3, max_rows: int = 3000) -> pd.DataFrame:
    pipe = out["models"][out["champion"]]
    Xte, yte = out["X_test"], out["y_test"]
    if len(Xte) > max_rows:
        idx = Xte.sample(max_rows, random_state=RANDOM_STATE).index
        Xte, yte = Xte.loc[idx], yte.loc[idx]
    pi = permutation_importance(pipe, Xte, yte, scoring="average_precision", n_repeats=n_repeats,
                                random_state=RANDOM_STATE, n_jobs=-1)
    df = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean, "std": pi.importances_std})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def playbook(out: dict, imp: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """For each top feature: how the top-grossing rate changes across its range."""
    X = pd.concat([out["X_train"], out["X_test"]])
    y = pd.concat([out["y_train"], out["y_test"]])
    rows = []
    for f in imp["feature"].head(top_n):
        s = X[f]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 4:
            try:
                q = pd.qcut(s, 4, duplicates="drop")
            except ValueError:
                continue
            rate = y.groupby(q, observed=True).mean()
            lo, hi = rate.iloc[0], rate.iloc[-1]
            rows.append({"lever": f, "low_group_rate": lo, "high_group_rate": hi,
                         "direction": "higher is better" if hi > lo else "lower is better",
                         "detail": f"top-grossing rate goes from {lo:.1%} (lowest quartile) to {hi:.1%} (highest quartile)"})
        else:
            rate = y.groupby(s.astype(str)).mean().sort_values()
            if len(rate) < 2:
                continue
            rows.append({"lever": f, "low_group_rate": rate.iloc[0], "high_group_rate": rate.iloc[-1],
                         "direction": f"best group: {rate.index[-1]}",
                         "detail": f"'{rate.index[-1]}' = {rate.iloc[-1]:.1%} vs '{rate.index[0]}' = {rate.iloc[0]:.1%}"})
    return pd.DataFrame(rows)


def tree_rules(out: dict, max_depth: int = 3) -> str | None:
    """Readable rules from a small, separately fitted decision tree."""
    X, y = out["X_train"], out["y_train"]
    prep = _preprocessor(X)
    Xt = prep.fit_transform(X)
    names = list(prep.transformers_[0][2]) + list(prep.transformers_[1][2])
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=30, class_weight="balanced",
                                  random_state=RANDOM_STATE).fit(Xt, y)
    return export_text(tree, feature_names=names)


def score_frame(out: dict, thr: float) -> pd.DataFrame:
    pipe = out["models"][out["champion"]]
    Xte = out["X_test"]
    proba = pipe.predict_proba(Xte)[:, 1]
    df = Xte.copy()
    df.insert(0, "actual_top_grossing", out["y_test"].values)
    df.insert(1, "top_grossing_probability", proba.round(4))
    df.insert(2, "recommended_action", np.where(proba >= thr, "BACK (invest in UA)", "HOLD"))
    return df.sort_values("top_grossing_probability", ascending=False)
