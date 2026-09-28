"""
Cric Auction IQ — Training Pipeline
EDA + Feature Engineering + ML Model + SHAP, on real, independently-sourced data.

Two things this pipeline deliberately does NOT do, and why:

1. It does not fill missing stats with a league average. A player who has
   never bowled has no economy rate to guess at — leaving it NaN (RandomForest
   handles NaN splits natively) plus an explicit `_missing` flag lets the
   model treat "never bowled" as a real, distinct signal instead of a fiction.

2. It does not train a sold/unsold classifier. Published IPL auction results
   only list players who were BOUGHT — there is no complete, structured list
   of who went unsold in any year. Building a classifier would mean
   inventing negative-class rows, which isn't something this project does.
   See docs/DATA_QUALITY.md.

The headline R² reported here is 5-fold cross-validated, grouped so the same
player can never appear in both the train and test fold of a given split —
without grouping, a player's other auction years leak into training and the
score is optimistic. This is a real methodological choice, not a default.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import joblib, os, json

from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "."
DATA_PATH   = f"{BASE}/data/ipl_auction_data.csv"
MODEL_DIR   = f"{BASE}/models"
PLOTS_DIR   = f"{BASE}/notebooks/plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "blue":   "#378ADD",
    "green":  "#1D9E75",
    "amber":  "#EF9F27",
    "coral":  "#D85A30",
    "purple": "#7F77DD",
    "gray":   "#888780",
}

# ─── 1. LOAD & INSPECT ────────────────────────────────────────────────────────
print("=" * 60)
print("  IPL AUCTION INTELLIGENCE — TRAINING PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"    Years: {df['year'].min()} - {df['year'].max()}")
print(f"    Players: {df['playername'].nunique()} unique")
print(f"    Franchises: {df['franchise'].nunique()}")

# ─── 2. EDA PLOTS ─────────────────────────────────────────────────────────────
print("\n[2] Generating EDA plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
fig.suptitle("Cric Auction IQ — Exploratory Data Analysis", fontsize=16, fontweight='bold', y=0.98)

ax = axes[0, 0]
yearly = df.groupby('year')['soldpricecr'].agg(['mean', 'median']).reset_index()
ax.plot(yearly['year'], yearly['mean'], marker='o', color=COLORS['blue'], linewidth=2, label='Mean price')
ax.plot(yearly['year'], yearly['median'], marker='s', color=COLORS['coral'], linewidth=2, linestyle='--', label='Median price')
ax.fill_between(yearly['year'], yearly['median'], yearly['mean'], alpha=0.15, color=COLORS['blue'])
ax.set_title("Avg Auction Price Trend (Rs Cr)", fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Price (Rs Cr)")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
role_colors = {"Batsman": COLORS['blue'], "Bowler": COLORS['green'],
               "All-Rounder": COLORS['amber'], "Wicketkeeper-Batsman": COLORS['purple']}
for role, color in role_colors.items():
    data = df[df['role'] == role]['soldpricecr']
    ax.hist(data, bins=20, alpha=0.6, color=color, label=role, edgecolor='white')
ax.set_title("Price Distribution by Role", fontweight='bold')
ax.set_xlabel("Sold Price (Rs Cr)"); ax.set_ylabel("Count")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
franchise_spend = df.groupby('franchise')['soldpricecr'].mean().sort_values(ascending=True).tail(10)
bars = ax.barh(franchise_spend.index, franchise_spend.values, color=COLORS['purple'], alpha=0.8)
ax.set_title("Avg Price Paid per Player by Franchise", fontweight='bold')
ax.set_xlabel("Avg Price (Rs Cr)")
for bar, val in zip(bars, franchise_spend.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'Rs{val:.1f}Cr', va='center', fontsize=8)
ax.grid(alpha=0.3, axis='x')

ax = axes[1, 1]
for role, color in role_colors.items():
    mask = df['role'] == role
    ax.scatter(df[mask]['career_matches'], df[mask]['soldpricecr'], c=color, alpha=0.5, s=40, label=role)
ax.set_title("Career Matches vs Sold Price", fontweight='bold')
ax.set_xlabel("Career Matches (at auction time)"); ax.set_ylabel("Sold Price (Rs Cr)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/eda_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: eda_overview.png")

fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
top_value = df.dropna(subset=['valueindex']).nlargest(15, 'valueindex')[
    ['playername', 'valueindex', 'soldpricecr', 'role']].reset_index(drop=True)
colors_list = [role_colors[r] for r in top_value['role']]
bars = ax.bar(top_value['playername'], top_value['valueindex'], color=colors_list, alpha=0.85)
ax.set_title("Top 15 — Best Value Players (Performance Score / Price)", fontsize=13, fontweight='bold')
ax.set_ylabel("Value Index")
plt.xticks(rotation=35, ha='right', fontsize=9)
handles = [mpatches.Patch(color=c, label=r) for r, c in role_colors.items()]
ax.legend(handles=handles, fontsize=8)
ax.grid(alpha=0.3, axis='y')
for bar, row in zip(bars, top_value.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'Rs{row.soldpricecr}Cr', ha='center', fontsize=7.5)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/value_index.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: value_index.png")

# ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[3] Feature engineering...")

le_role = LabelEncoder()
le_nat  = LabelEncoder()
df['role_enc'] = le_role.fit_transform(df['role'])
df['nationality_enc'] = le_nat.fit_transform(df['nationality'])

joblib.dump(le_role, f"{MODEL_DIR}/le_role.pkl")
joblib.dump(le_nat,  f"{MODEL_DIR}/le_nat.pkl")

FEATURES = [
    'role_enc', 'nationality_enc', 'year', 'career_matches',
    'battingavg', 'battingsr', 'runsscored', 'fours', 'sixes',
    'wickets', 'economyrate', 'bowlingavg', 'bowlingsr',
    'catches', 'stumpings', 'basepricecr',
    'battingavg_missing', 'battingsr_missing', 'economyrate_missing', 'bowlingavg_missing',
]
TARGET = 'soldpricecr'

X = df[FEATURES]
y = df[TARGET]
groups = df['playername']

with open(f"{MODEL_DIR}/feature_names.json", 'w') as f:
    json.dump(FEATURES, f)

print(f"    Features: {len(FEATURES)} | Rows: {len(X)} | Unique players: {groups.nunique()}")

# ─── 4. TRAIN REGRESSION MODEL ────────────────────────────────────────────────
print("\n[4] Training price prediction model (Random Forest)...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [8, 12, None],
    'min_samples_leaf': [2, 4]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Holdout metrics (single 20% split — noisy at this sample size, reported for reference)
y_pred = best_rf.predict(X_test)
holdout_r2   = r2_score(y_test, y_pred)
holdout_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
holdout_mae  = mean_absolute_error(y_test, y_pred)

# Headline metric: 5-fold CV over the WHOLE dataset, grouped by player so the
# same player's other seasons never leak from test into train.
gkf = GroupKFold(n_splits=5)
cv_model = RandomForestRegressor(**grid_search.best_params_, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(cv_model, X, y, groups=groups, cv=gkf, scoring='r2', n_jobs=-1)
headline_r2 = float(cv_scores.mean())
headline_r2_std = float(cv_scores.std())

print(f"    Best params: {grid_search.best_params_}")
print(f"    Headline R² (5-fold grouped CV) = {headline_r2:.4f} (+/- {headline_r2_std:.4f})")
print(f"    Holdout R² (single 20% split)   = {holdout_r2:.4f}")
print(f"    Holdout RMSE = Rs{holdout_rmse:.2f} Cr | Holdout MAE = Rs{holdout_mae:.2f} Cr")

# Final model used by the app is refit on the full training split (not full
# dataset) so the held-out test rows stay genuinely unseen for the plot below.
joblib.dump(best_rf, f"{MODEL_DIR}/price_predictor.pkl")
metrics = {
    "headline_r2_grouped_cv_mean": round(headline_r2, 4),
    "headline_r2_grouped_cv_std": round(headline_r2_std, 4),
    "headline_r2_note": "5-fold CV over the whole dataset, folds grouped by player so a player's other auction years cannot leak into training. This is the number to quote.",
    "holdout_r2": round(holdout_r2, 4),
    "holdout_rmse_cr": round(holdout_rmse, 4),
    "holdout_mae_cr": round(holdout_mae, 4),
    "holdout_note": "single 20% random split, ungrouped — noisier and slightly optimistic vs. the headline; kept for the Actual-vs-Predicted plot only",
    "best_params": grid_search.best_params_,
    "n_rows": int(len(df)),
    "n_players": int(groups.nunique()),
}
with open(f"{MODEL_DIR}/regression_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
ax.scatter(y_test, y_pred, alpha=0.5, color=COLORS['blue'], edgecolors='white', s=60)
lims = [0, max(y_test.max(), y_pred.max()) + 1]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel("Actual Price (Rs Cr)"); ax.set_ylabel("Predicted Price (Rs Cr)")
ax.set_title(f"Actual vs Predicted (holdout) | R2={holdout_r2:.3f} | Headline grouped-CV R2={headline_r2:.3f}", fontweight='bold', fontsize=11)
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: actual_vs_predicted.png")

# ─── 5. SHAP EXPLAINABILITY ───────────────────────────────────────────────────
print("\n[5] Computing SHAP values...")

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
mean_shap = np.abs(shap_values).mean(axis=0)
feat_importance = pd.Series(mean_shap, index=FEATURES).sort_values(ascending=True)
colors_shap = [COLORS['coral'] if v > feat_importance.median() else COLORS['blue']
               for v in feat_importance.values]
bars = ax.barh(feat_importance.index, feat_importance.values, color=colors_shap, alpha=0.85)
ax.set_title("SHAP Feature Importance — What Drives Auction Price?", fontsize=13, fontweight='bold')
ax.set_xlabel("Mean |SHAP Value| (Rs Cr impact)")
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/shap_global.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: shap_global.png")
print(f"    Top 5 features by mean |SHAP|:\n{feat_importance.sort_values(ascending=False).head(5)}")

joblib.dump(explainer, f"{MODEL_DIR}/shap_explainer.pkl")

# ─── 6. SOLD/UNSOLD CLASSIFIER — DELIBERATELY NOT BUILT ───────────────────────
print("\n[6] Sold/unsold classifier: skipped.")
clf_metrics = {
    "skipped": True,
    "reason": "Published IPL auction sources list only the players who were bought. There is no complete, structured record of who went unsold in a given year, so a real negative class doesn't exist in the data. Training this would require fabricating unsold rows, which this project does not do. See docs/DATA_QUALITY.md."
}
with open(f"{MODEL_DIR}/classifier_metrics.json", 'w') as f:
    json.dump(clf_metrics, f, indent=2)

# ─── 7. SAVE SUMMARY STATS FOR DASHBOARD ──────────────────────────────────────
print("\n[7] Saving dashboard data...")

summary = {
    "total_records": len(df),
    "unique_players": int(df['playername'].nunique()),
    "avg_price_overall": round(df['soldpricecr'].mean(), 2),
    "max_price": round(df['soldpricecr'].max(), 2),
    "min_price": round(df['soldpricecr'].min(), 2),
    "years": sorted(df['year'].unique().tolist()),
    "franchises": sorted(df['current_franchise'].unique().tolist()),
    "roles": sorted(df['role'].unique().tolist()),
}
with open(f"{MODEL_DIR}/summary_stats.json", 'w') as f:
    json.dump(summary, f, indent=2)

top_val = df.dropna(subset=['valueindex']).nlargest(20, 'valueindex')[
    ['playername', 'role', 'franchise', 'year', 'soldpricecr', 'performancescore', 'valueindex']
].reset_index(drop=True)
top_val.to_csv(f"{MODEL_DIR}/top_value_players.csv", index=False)

fran_eff = df.groupby('current_franchise').agg(
    total_spend=('soldpricecr', 'sum'),
    avg_price=('soldpricecr', 'mean'),
    avg_value_index=('valueindex', 'mean'),
    players_bought=('playername', 'count'),
    avg_performance=('performancescore', 'mean')
).round(2).reset_index().rename(columns={'current_franchise': 'franchise'})
fran_eff.to_csv(f"{MODEL_DIR}/franchise_efficiency.csv", index=False)

df.to_csv(f"{MODEL_DIR}/processed_data.csv", index=False)

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print(f"  Models saved  -> {MODEL_DIR}/")
print(f"  Plots saved   -> {PLOTS_DIR}/")
print(f"\n  Key metrics:")
print(f"    Headline R² (grouped 5-fold CV) = {headline_r2:.4f}")
print(f"    Holdout R² (20% split)          = {holdout_r2:.4f}")
print(f"    Sold/unsold classifier          = not built (see classifier_metrics.json)")
print(f"\n  Next step: streamlit run streamlit_app/app.py")
