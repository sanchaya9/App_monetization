"""Streamlit dashboard: What makes a mobile app top-grossing?

Run:  streamlit run app.py
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="App Monetization Dashboard", page_icon="📱", layout="wide")


# ----------------------------------------------------------------------------
# Cached helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def demo_df() -> pd.DataFrame:
    return pl.make_demo_data()


@st.cache_resource(show_spinner=False)
def train(df: pd.DataFrame, target: str, drop_cols: tuple, top_pct: float, tune: bool):
    X, y, desc = pl.prepare(df, target, list(drop_cols), top_pct)
    out = pl.run_models(X, y, tune=tune)
    out["label_desc"] = desc
    out["imp"] = pl.feature_importance(out)
    out["play"] = pl.playbook(out, out["imp"])
    out["rules"] = pl.tree_rules(out)
    out["metrics"] = pl.metrics_table(out)
    out["base_rate"] = float(y.mean())
    out["n_rows"] = len(y)
    return out


# ----------------------------------------------------------------------------
# Sidebar: data + scenario
# ----------------------------------------------------------------------------
st.title("📱 What makes a mobile app top-grossing?")

with st.sidebar:
    st.header("1. Data")
    up = st.file_uploader("Upload the Kaggle CSV", type=["csv"])
    use_demo = st.checkbox("Use synthetic demo data instead", value=up is None,
                           help="Made-up data so you can try the dashboard. NOT the Kaggle dataset.")

if up is None and not use_demo:
    st.info("Upload the CSV in the sidebar, or tick **Use synthetic demo data** to explore the dashboard.")
    st.stop()

df = demo_df() if (up is None) else read_csv(up.getvalue())
if up is None:
    st.warning("Showing **synthetic demo data** - all numbers below are illustrative, not real Kaggle results.")

with st.sidebar:
    cols = list(df.columns)
    target = st.selectbox("Target column", cols, index=cols.index(pl.guess_target(df)),
                          help="0/1 or Yes/No top-grossing flag, or a revenue column.")
    top_pct = 10.0
    is_flag = pl._is_binary_like(df[target])
    if not is_flag:
        top_pct = st.slider("Revenue column -> label top X% as 'top-grossing'", 1, 50, 10)
    default_drop = sorted(set(pl.guess_id_columns(df, target) + pl.guess_leak_columns(df, target)))
    drop_cols = st.multiselect("Columns to drop (IDs / revenue leakage)", [c for c in cols if c != target],
                               default=default_drop)
    tune = st.checkbox("Tune hyperparameters (slower)", value=False)

    st.header("2. Business scenario")
    cost = st.number_input("Cost to back one app (UA spend)", min_value=0.0, value=5.0, step=1.0)
    value = st.number_input("Value of capturing one top-grosser", min_value=0.0, value=100.0, step=10.0)
    budget = st.number_input("Total marketing budget", min_value=0.0, value=5000.0, step=500.0)

with st.spinner("Training Decision Tree, Random Forest and boosted model..."):
    try:
        out = train(df, target, tuple(drop_cols), float(top_pct), tune)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not train on this data: {e}")
        st.stop()

res, champ = out["results"], out["champion"]
yte = out["y_test"].values
proba = res[champ]["proba"]
sweep = pl.threshold_sweep(yte, proba, cost, value, budget)
best = pl.best_threshold(sweep)

with st.sidebar:
    st.header("3. Decision threshold")
    mode = st.radio("Threshold", ["Auto (best net value within budget)", "Manual"])
    thr = float(best["threshold"]) if mode.startswith("Auto") else st.slider("Probability cut-off", 0.01, 0.99, 0.5, 0.01)
    if not pl.HAS_XGB:
        st.caption("xgboost not installed: a scikit-learn gradient-boosting model is used as a stand-in. "
                   "`pip install xgboost` for the real thing.")

cur = sweep.iloc[(sweep["threshold"] - thr).abs().argmin()]

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
t_over, t_models, t_thr, t_levers, t_score, t_data = st.tabs(
    ["Overview", "Model comparison", "Threshold & budget", "Levers (playbook)", "Score an app", "Data & export"])

with t_over:
    st.caption(out["label_desc"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Apps analysed", f"{out['n_rows']:,}")
    c2.metric("Top-grossing share", f"{out['base_rate']:.1%}")
    c3.metric("Champion model", champ)
    c4.metric("Champion PR-AUC", f"{res[champ]['pr_auc']:.3f}")
    st.subheader(f"At threshold {thr:.2f} (champion model, held-out test set)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Apps to back", f"{int(cur['apps_backed']):,}")
    k2.metric("Hits captured", f"{int(cur['hits_captured']):,}", f"{cur['recall']:.1%} of all hits")
    k3.metric("Wasted spend on", f"{int(cur['wasted']):,} apps")
    k4.metric("Spend", f"{cur['spend']:,.0f}")
    k5.metric("Net value", f"{cur['net_value']:,.0f}")
    if not cur["within_budget"]:
        st.error("This threshold needs more spend than your budget. Raise the threshold or the budget.")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    top = out["imp"].head(10).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#2a6fdb")
    ax.set_xlabel("Permutation importance (drop in PR-AUC)")
    ax.set_title("Top levers")
    fig.tight_layout()
    st.pyplot(fig)

with t_models:
    st.subheader("Evaluation metrics (held-out test set)")
    st.dataframe(out["metrics"].style.format({c: "{:.3f}" for c in
                 ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]}), use_container_width=True)
    st.caption(f"Champion = highest PR-AUC: **{champ}**. PR-AUC is used because top-grossers are rare.")
    if res[champ]["best_params"]:
        st.json({k.replace("clf__", ""): v for k, v in res[champ]["best_params"].items()})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in res.items():
        a1.plot(*r["roc_curve"], label=f"{name} (AUC {r['roc_auc']:.3f})")
        a2.plot(r["pr_curve"][1], r["pr_curve"][0], label=f"{name} (AP {r['pr_auc']:.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    a2.axhline(out["base_rate"], color="k", ls="--", lw=0.8)
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.subheader(f"Confusion matrix - {champ} at threshold {thr:.2f}")
    st.dataframe(pl.confusion_df(yte, proba, thr), use_container_width=True)

with t_thr:
    st.subheader("Cost-sensitive threshold & budget")
    st.write(f"Best threshold within budget: **{best['threshold']:.2f}** -> back **{int(best['apps_backed']):,}** apps, "
             f"capture **{int(best['hits_captured']):,}** hits, net value **{best['net_value']:,.0f}**.")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(sweep["threshold"], sweep["precision"], label="Precision")
    a1.plot(sweep["threshold"], sweep["recall"], label="Recall")
    a1.axvline(thr, color="red", ls="--", label=f"Chosen {thr:.2f}")
    a1.set(xlabel="Threshold", title="Precision vs recall")
    a1.legend()
    a2.plot(sweep["threshold"], sweep["net_value"], color="green")
    feas = sweep[sweep["within_budget"]]
    if not feas.empty:
        a2.scatter(feas["threshold"], feas["net_value"], s=8, color="green", label="Within budget")
    a2.axvline(thr, color="red", ls="--")
    a2.set(xlabel="Threshold", ylabel="Net value", title="Net value (hits x value - spend)")
    a2.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.dataframe(sweep.iloc[::5].reset_index(drop=True), use_container_width=True)

with t_levers:
    st.subheader("Monetization playbook")
    imp = out["imp"]
    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2a6fdb")
    ax.set_xlabel("Permutation importance (drop in PR-AUC)")
    fig.tight_layout()
    st.pyplot(fig)
    for _, r in out["play"].iterrows():
        st.markdown(f"- **{r['lever']}** ({r['direction']}): {r['detail']}")
    with st.expander("Decision-tree rules (depth 3)"):
        st.code(out["rules"])
    st.caption("These are associations in the data, not proof that changing a lever causes revenue.")

with t_score:
    st.subheader("Score a new app")
    X0 = out["X_train"]
    pipe = out["models"][champ]
    vals = {}
    form_cols = st.columns(3)
    with st.form("score_form"):
        for i, c in enumerate(out["feature_names"]):
            box = form_cols[i % 3]
            if pd.api.types.is_numeric_dtype(X0[c]):
                vals[c] = box.number_input(c, value=float(X0[c].median()))
            else:
                opts = sorted(X0[c].dropna().astype(str).unique().tolist())[:200]
                vals[c] = box.selectbox(c, opts)
        go = st.form_submit_button("Score")
    if go:
        row = pd.DataFrame([vals])[out["feature_names"]]
        for c in out["feature_names"]:
            if pd.api.types.is_numeric_dtype(X0[c]):
                row[c] = pd.to_numeric(row[c])
        p = float(pipe.predict_proba(row)[:, 1][0])
        action = "BACK (invest in UA)" if p >= thr else "HOLD"
        st.metric("Top-grossing probability", f"{p:.1%}")
        (st.success if p >= thr else st.warning)(f"Recommendation at threshold {thr:.2f}: **{action}**")

with t_data:
    st.subheader("Scored test apps")
    scored = pl.score_frame(out, thr)
    st.dataframe(scored.head(500), use_container_width=True)
    st.download_button("Download scored_apps_test.csv", scored.to_csv(index=False).encode(), "scored_apps_test.csv", "text/csv")
    st.download_button("Download metrics.csv", out["metrics"].to_csv(index=False).encode(), "metrics.csv", "text/csv")
    st.download_button("Download threshold_sweep.csv", sweep.to_csv(index=False).encode(), "threshold_sweep.csv", "text/csv")
    with st.expander("Raw data preview"):
        st.dataframe(df.head(100), use_container_width=True)
