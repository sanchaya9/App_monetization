# App Monetization Streamlit Dashboard

Interactive dashboard for the Kaggle dataset "What makes a mobile app top-grossing (10,000 apps)".

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Then upload the Kaggle CSV in the sidebar (or tick "Use synthetic demo data" to try it first).

## What you see
- **Overview** - KPIs, champion model, apps to back, hits captured, net value, top levers chart
- **Model comparison** - Decision Tree vs Random Forest vs XGBoost: metrics table, ROC and PR curves, confusion matrix
- **Threshold & budget** - precision/recall vs threshold and net-value curve; sliders for cost, value, budget
- **Levers (playbook)** - feature importance chart, how each lever moves the top-grossing rate, decision-tree rules
- **Score an app** - form to enter a new app's features and get a probability and BACK/HOLD recommendation
- **Data & export** - scored test apps and metrics as downloadable CSVs

## Notes
- Pick the target column in the sidebar. A 0/1 or Yes/No flag is used as-is; a revenue column is turned into a "top X%" label.
- Revenue-like and ID columns are pre-selected to drop (avoids label leakage); you can change them.
- If xgboost is not installed, a scikit-learn gradient-boosting model is used instead.
- Files: `app.py` (UI), `pipeline.py` (ML logic).
