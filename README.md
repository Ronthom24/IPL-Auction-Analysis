# IPL Auction Intelligence

A Business Intelligence project on 18 seasons of real IPL auction data: predicts
auction prices with an explainable ML model, surfaces value-for-money picks,
and ranks franchise spend efficiency — through an interactive Streamlit
dashboard.

**Every number below is real and reproducible from this repo.** No stat in
the dataset is synthetic or derived from price — see
[`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md) for exactly what's real, what's
missing, and the leakage bug this project's earlier version had.

---

## What this actually is

| | |
|---|---|
| **Data** | 941 real IPL auction records, 2009–2026, 463 unique players, 15 franchises |
| **Features** | Real per-season player stats computed from Cricsheet ball-by-ball data (batting average, strike rate, wickets, economy, etc.) — independent of auction price |
| **Model** | Random Forest Regressor · headline **R² = 0.33** (5-fold CV, grouped by player) |
| **Explainability** | SHAP (TreeExplainer) — per-prediction and global feature importance |
| **App** | 5-page Streamlit dashboard: Overview, Player Analysis, Value Finder, Price Predictor, Franchise Insights |
| **Not included** | A sold/unsold classifier — no real "unsold" ground truth exists in published auction data (see below) |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Build the dataset from the real source data (data/external/)
python data/build_dataset.py

# 2. Train the model + generate plots
python notebooks/train_models.py

# 3. Launch the dashboard
streamlit run streamlit_app/app.py
```

---

## Project structure

```
ipl_auction_analysis/
├── data/
│   ├── external/                    ← real source data (see Data Provenance)
│   │   ├── cricsheet_training_dataset.csv
│   │   └── ipl_auction_raw.csv
│   ├── build_dataset.py             ← builds the model-ready dataset
│   └── ipl_auction_data.csv         ← output (941 rows)
│
├── notebooks/
│   ├── train_models.py              ← EDA + training + SHAP pipeline
│   └── plots/                       ← EDA charts (PNG)
│
├── models/                          ← saved model + dashboard data (auto-generated)
│   ├── price_predictor.pkl
│   ├── shap_explainer.pkl
│   ├── regression_metrics.json      ← the real, current metrics
│   ├── classifier_metrics.json      ← explains why there's no classifier
│   └── *.csv, *.json
│
├── streamlit_app/
│   └── app.py                       ← 5-page dashboard
│
├── powerbi_guide/
│   └── POWERBI_SETUP.md
│
├── docs/
│   └── DATA_QUALITY.md              ← known issues, read before trusting a number
│
├── report/
│   └── PROJECT_REPORT.md
│
└── requirements.txt
```

---

## Data provenance

The source files in `data/external/` are real IPL auction prices (from
published auction records) joined to real per-season player statistics
computed from **Cricsheet** ball-by-ball match data. They were produced by a
data pipeline originally built for a related project of mine (a cricket
scores/valuation platform); this project re-derives its own model-ready
dataset and its own performance-score formula from that already-audited real
data, rather than re-scraping from scratch. Full transparency on this because
it matters for how you read the "real data" claim: **the auction prices and
player stats are genuinely real and independently sourced — the specific
`build_dataset.py`/`train_models.py` pipeline and the Streamlit app in this
repo are original to this project.**

---

## Model performance (real numbers, from `models/regression_metrics.json`)

| Metric | Value | What it means |
|---|---|---|
| **Headline R²** | **0.3295** | 5-fold CV, grouped by player, over the full 941-row dataset. This is the number to quote. |
| Holdout R² | 0.3855 | Single 20% random split — noisier at this sample size, kept only for the actual-vs-predicted plot |
| Holdout MAE | ₹1.94 Cr | |
| Holdout RMSE | ₹3.03 Cr | |
| Best params | `n_estimators=400, max_depth=12, min_samples_leaf=4` | via 5-fold `GridSearchCV` |

**Why grouped CV matters:** a player who's been auctioned 3 times appears as
3 rows. An ungrouped split can put one of a player's years in training and
another in test — the model partly memorizes the player rather than learning
transferable price drivers. Grouping by player closes that leak. This
project's earlier version didn't do this *and* had a much bigger problem —
see the note below.

**Why R² = 0.33 and not higher:** IPL auction prices are driven by things no
stats table captures — bidding wars between two teams that both need the same
role, a marquee name commanding a premium beyond current form, a franchise's
specific squad gap that year. A third of the variance being explainable by
career batting/bowling numbers alone is a real, honest result for this kind
of noisy, sentiment-driven market — not a shortfall to apologize for.

### The leakage bug this project used to have

An earlier version of this dataset generated "performance stats" as a
mathematical function of each player's own historical auction price (higher
price → higher assumed batting average, by formula), then tried to predict
price from those stats. That's circular — it reported R² ≈ 0.64, which was
inflated by the leak, not genuine predictive skill. This version replaces
that entirely with real, independent, price-blind stats, and the honest R²
dropped to 0.33. That drop is the leakage being removed, not the model
getting worse — it's the actual price signal that real performance data
carries. This is worth explaining directly if asked, not glossing over.

---

## Why there's no sold/unsold classifier

Published IPL auction sources (year-by-year results tables) only list players
who were **bought**. There's no complete, structured record of who went
unsold in a given year — unsold players are occasionally mentioned in prose
for a handful of marquee names, never as a full per-year list. Building a
classifier here would mean inventing negative-class rows to train on, which
this project deliberately doesn't do. `models/classifier_metrics.json`
records this explicitly rather than shipping a broken or fabricated model.

---

## The Streamlit app — 5 pages

| Page | What it does |
|---|---|
| 📊 Overview | KPIs, price trend, role distribution, franchise spend, nationality mix |
| 🔍 Player Analysis | Filterable scatter plots, heatmaps, full data table |
| 💡 Value Finder | Value-index ranking + a greedy budget-constrained squad optimizer |
| 🤖 Price Predictor | Enter a player profile → predicted price + SHAP waterfall explanation |
| 🏢 Franchise Insights | Spend efficiency, rankings, franchise-level stats |

Franchises that were renamed mid-history (Delhi Daredevils→Capitals, Kings XI
Punjab→Punjab Kings, RCB Bangalore→Bengaluru) are merged under their current
identity for aggregation, while the underlying player-record table keeps the
name in force at the time — a deliberate choice, documented in the app.

---

## Honest limitations (worth knowing before an interview, not just a demo)

- **Value Index is a heuristic, not a validated signal.** It's
  `performance_score ÷ sold_price_cr` using this project's own scoring
  formula — there's no outcome data (next-season performance, retention,
  etc.) checking whether the "best value" picks actually paid off. The app
  says this explicitly on the Value Finder page.
- **The budget optimizer is greedy, not optimal.** It sorts by value index
  and fills role quotas under a cap — a fast approximation, not a knapsack/ILP
  solve. It can miss combinations a true optimal solver would find.
- **Missing stats are real, not filled.** A pure bowler has no batting
  average on record — it's left as `NaN` with an explicit `_missing` flag,
  not a fabricated league-average substitute. See `docs/DATA_QUALITY.md`.
- **No deployment yet.** This runs locally (`streamlit run`). If you deploy
  it (e.g. Streamlit Community Cloud), update this section with the real URL
  — don't claim a live link that doesn't exist.
- **No CI/CD, no automated tests.** This is a data science / BI project, not
  a production service — say so plainly if asked.

---

## Data sources & attribution

- Ball-by-ball match data: [Cricsheet](https://cricsheet.org) (free, publicly
  published cricket data — used here to compute real per-season player stats)
- Auction prices: published IPL auction result records

---

## Tools & technologies

Python · pandas · NumPy · scikit-learn (RandomForest, GridSearchCV,
GroupKFold) · SHAP · Streamlit · Plotly · Power BI (guide) · joblib

---

## Viva / interview prep

**Q: Why Random Forest and not XGBoost/gradient boosting?**
RF gives comparable accuracy with more transparent interpretability via
`shap.TreeExplainer`, and is less prone to overfitting on ~940 rows without
careful tuning.

**Q: Your R² is 0.33 — is that good?**
For a market this driven by bidding psychology and franchise-specific need,
yes — it means real, independently-measured performance stats explain about a
third of price variance with no leakage. See "Why R² = 0.33" above for the
full answer, including the leakage bug this replaced.

**Q: What does SHAP tell you here?**
It assigns each feature a ₹-crore contribution to a specific prediction. For
example, a low IPL base price is consistently the single biggest downward
driver in this dataset — because BCCI's own base-price tier is a strong prior
on how the market will bid.

**Q: How would you improve this with more time?**
(1) A real outcome-based check for the Value Index — e.g. did "value" picks
outperform in the following season; (2) an actual optimal-solver budget
builder (ILP) instead of greedy; (3) live CricAPI/Cricbuzz integration for
in-season updates; (4) a real, sourced unsold-players list (year-by-year
recap articles) to finally make sold/unsold classification honest.
